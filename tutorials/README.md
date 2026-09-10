# MiniVLA 学习笔记

从零手写一个迷你 VLA（Vision-Language-Action），理解 Alpamayo 1.5 的核心机制。

> 适合：熟悉 3D 检测 / BEV、有一定 LLM 基础、想系统理解 VLA 的读者。
> 每个 stage 是一个可独立运行的 `.py`，从框架到训练逐步递进。

---

## 一、什么是 VLA

VLA = **V**ision + **L**anguage + **A**ction。Alpamayo 里：

- **Vision**：多相机画面（4~8 路）
- **Language**：因果链推理（Chain-of-Causation）+ 导航指令
- **Action**：未来轨迹（64 个 waypoints，6.4s @ 10Hz），不是油门/方向盘

**一句话心智模型**：

> VLM 看图 + 历史轨迹 → 写一段推理 → 把推理后的隐状态（KV cache）当作条件，喂给一个 diffusion 专家去「去噪」出未来轨迹。

---

## 二、整体架构（一图流）

```
输入
├─ 历史轨迹序列 (B, H, 2)
└─ （真实里还有图片，toy 省略）

  ① 条件 CONDITION
     历史轨迹 ──► ConditionEncoder ──► condition (B, H, 64)

  ② 扩散去噪 DIFFUSION（循环 N 步）
     噪声 x (B, 64, 2)  +  时间 t (B,)
       └─► ActionInProj ──► (B, 64, 64) "token embedding"
       └─► Expert ──(cross-attn 读 condition)──► (B, 64, 64)
       └─► ActionOutProj ──► 向量场 v (B, 64, 2)
       x ← x + dt·v

  ③ 动作→轨迹
     x (B, 64, 2) ──► ActionSpace.action_to_traj ──► (B, 64, 3)
```

三个数字含义（别混）：
- **64（中间维）** = 未来 waypoint 数（时间轴）
- **64（最后维）** = HIDDEN 特征维（特征轴）
- 两者是**巧合同值**，无关。

---

## 三、16 个 Stage 回顾

| Stage | 文件 | 核心概念 | 学到什么 |
|---|---|---|---|
| 0 | `stage0_framework.py` | 模块接口 + 数据流 | 6 个模块 + `sample` 三行（条件→扩散→转轨迹），shape 自检 |
| 1 | `stage1_action_space.py` | 单轮车运动学 | 动作 = (加速度, 曲率)，积分成轨迹，物理合理 + 平滑 |
| 2 | `stage2_flow_matching.py` | 扩散采样 | 从噪声出发，每步问向量场「往哪走」，欧拉积分 |
| 3 | `stage3_projections.py` | Fourier 编码 | 动作/时间编码成 embedding，进 transformer 的「桥」 |
| 4 | `stage4_expert.py` | cross-attention | 动作当 query 去「查」条件，VLA 的灵魂 |
| 5 | `stage5_assembly.py` | 组装 | 把 1~4 拼成完整 MiniVLA，跑通闭环 |
| 6 | `stage6_training.py` | flow matching 训练 | 预测向量场 v = x1 - x0，离散 condition 控制轨迹 |
| 7 | `stage7_history_condition.py` | 序列编码条件 | condition 从历史轨迹「读」出来，逼近真实 VLM |
| 8 | `stage8_multimodal.py` | 多模态（并列分支，非 stage7 延续） | 同一条件采出左转/右转两簇（κ 0.05→0.2 让两模式分得开），回归做不到 |
| 9 | `stage9_vision.py` | 视觉编码（真 ViT） | transformers 的 ViT 把图片编码成 visual tokens 当 condition |
| 10 | `stage10_text.py` | 文本编码 | 文本 token → embedding → transformer → condition |
| 11 | `stage11_fusion.py` | 多模态融合（收官） | 历史+图片+文本 三路 token concat 成一个 condition |
| 12 | `stage12_two_cameras.py` | 多相机（文本标签） | 共享 ViT + 文本标签标识相机 + concat，多相机怎么融 |
| 13 | `stage13_coc.py` | 自回归 CoC 生成 | 因果 transformer 自回归生成推理，隐状态当 condition（对齐真实 VLM） |
| 14 | `stage14_complete.py` | 完整输入 + CoC | 历史 + 图片一起进 CosmosReason，最接近真实 Alpamayo 输入侧 |
| 15 | `stage15_cfg.py` | CFG 引导 | 引导权重 w 放大条件：`v=(1-w)·v_uncond + w·v_cond`，w>1 外推 |

**演进脉络（每个 stage 相对上一个改了什么）：**

| 过渡 | 改了什么 |
|---|---|
| 0→5 | 自包含、逐个搭模块（接口→动作空间→扩散→投影→Expert→组装） |
| 5→6 | FlowMatching 的 batch 参数化（推理固定 batch → 训练可变 batch）；加训练循环 |
| 6→7 | 并列分支：condition 从「离散 Embedding 查表」→「历史序列 → ConditionEncoder」 |
| 6→8 | 并列分支：做「多模态」demo（用固定 condition，κ 从 0.05 放大到 0.2 让两模式分得开） |
| 6→9 | 并列分支：补「V」，condition 从「真 ViT 编码图片」来 |
| 6→10 | 并列分支：补「L」，condition 从「文本 token」来 |
| 9/10→11 | 融合：三路 token（历史+图片+文本）concat 成一个 condition（收官） |
| 11→12 | 多相机：共享 ViT + 相机文本标签（「front/side」拼进视觉 token 前） |
| 12→13 | 自回归：因果 transformer 生成推理（teacher forcing 训练 + 贪心生成），隐状态当 condition |
| 13→14 | 补回历史输入：历史 + 图片融合进 CosmosReason 输入序列，再自回归生成 CoC |
| 14→15 | CFG：条件丢弃训练（50% 空条件）+ 引导权重 w 加权两个向量场 |

> 运行提示：`common.py` 是 stage4 之后抽出的**共享积木**（ActionSpace / FlowMatching / 投影 / Expert + 常量）；stage6~15 通过 `from common import ...` 复用，运行前需 `common.py` 在同目录。stage0~5 仍是自包含的（造积木阶段）。

---

## 四、关键概念速查

| 概念 | 一句话 |
|---|---|
| **condition** | VLM（或 encoder）输出的隐状态，编码了「图 + 历史 + 推理」，是 Expert 的 cross-attn 的 key/value |
| **向量场** | 给动作空间每个点配一个「去噪方向」，`step_fn(x,t)` 算当前点的方向 |
| **flow matching** | 学一个向量场把噪声「推」向数据；训练目标 v = x1 - x0 |
| **cross-attention** | query=动作 token，key/value=条件 token，让条件塑造动作 |
| **非因果 self-attn** | Expert 的动作 token 互相可见（轨迹是整体，不像 LLM 从左到右） |
| **action space** | 用 (accel, curvature) 而非原始 xyz，物理合理、低维、平滑 |
| **Fourier 编码** | 标量 → 多频率 sin/cos，让 MLP 能表示高频函数（同 NeRF 位置编码） |
| **temperature** | 缩放初始噪声；低 → 稳定但少多样性；**训练和推理要一致** |
| **CFG** | 引导权重 w 放大条件：`v=(1-w)·v_uncond + w·v_cond`，w>1 外推出训练数据没有的更强遵循 |

---

## 五、踩过的坑（实战经验）

1. **小信号放大**：目标动作 κ=±0.05 埋在 σ=1 的噪声里，残差 loss 0.01 在 κ 上就是 ~0.003 误差，积分 6.4s 放大成 2~3m 的横向偏移。轨迹对曲率极其敏感。

2. **temperature 要训练/推理一致**：模型在 temp=1.0 下训练，推理降 temp=0.3 是「分布漂移」，去噪反而更差。低温度稳定采样要在**训练阶段**就定好。

3. **CPU vs GPU 看模型规模**：toy 模型 ~25 万参数，CPU 秒级跑完；GPU 反而因 kernel 启动开销更快不了。真正的 8B 模型才需要 GPU（24GB 显存）。

---

## 六、与真实 Alpamayo 的对照

| Toy 模块 | 真实模块 | 真实文件 |
|---|---|---|
| ActionSpace | `UnicycleAccelCurvatureActionSpace` | `action_space/unicycle_accel_curvature.py` |
| FlowMatching | `FlowMatching` | `diffusion/flow_matching.py` |
| ActionInProj/OutProj | `PerWaypointActionInProjV2` / out_proj | `models/action_in_proj.py` |
| Expert | `self.expert`（Qwen3 text transformer，无 embed_tokens） | `models/alpamayo1_5.py` |
| ViT / CosmosReason（因果 transformer，自回归生成 CoC） | `self.vlm`（Qwen3-VL：ViT + LLM，看图自回归生成 CoC） | `models/base_model.py` |
| MiniVLA.sample | `sample_trajectories_from_data_with_vlm_rollout` | `models/alpamayo1_5.py:218` |

**toy 和真实的差距**：toy 用循环积分（真实用 cumsum 向量化 + 梯形积分）；toy 的 ViT/CosmosReason/Expert 都是小尺寸、随机初始化（真实是大的预训练模型）；toy 的导航指令是合成 embedding（真实是自然语言 nav 指令）。CFG 已实现（stage15）。

**关于「一个模型 vs 多个模块」**：真实代码里只有一个大模型 `self.vlm`（Qwen3-VL-8B），它**内部**包含三部分，对应 toy 的三个模块：

| Qwen3-VL 内部 | 干什么 | 对应 toy |
|---|---|---|
| Vision Encoder（ViT） | 图 → visual tokens | `build_vit`（stage9） |
| tokenizer + embed_tokens | 词 → 词向量 | `nn.Embedding`（即「Text Encoder」） |
| LLM 的 transformer 层 | 自回归生成 CoC | `CausalBlock` 堆（即「Cosmos Reason Backbone」） |

所以「Text Encoder」和「Cosmos Reason Backbone」**不是两个模型**，而是 Qwen3-VL 这一个模型内部的两部分；真正的第二个模型是 `self.expert`（Trajectory Decoder / 去噪器）。

---

## 七、VLA vs BEV：为什么不需要显式 3D

三个来自 BEV/UniAD 视角的常见疑问：

1. **多相机怎么知道谁是谁？** 靠**文本标签**——`helper.py` 在每路相机的图前插 `"Front camera: "` 等文本，prompt 是「文本标签 + 图片」交错排列。BEV 用外参硬编码几何，VLA 用语言描述传感器。

2. **外参不需要吗？** 推理时不显式给外参（输入只有图 + ego 历史 + 文本），但外参**隐式学进权重**——相机装固定位置，模型从训练数据学会「前相机画面对应车前方什么位置」。代价：换传感器布局要重训。

3. **不需要猜 3D/投影吗？** 对，无显式深度 / BEV / 投影，模型从 2D 图 + ego 历史**隐式**理解 3D。这是「端到端」vs「模块化 BEV」的分野：BEV 显式可解释、数据量要求低、但对标定敏感；VLA 端到端简单、免标定，但黑盒、需海量数据。

---

## 八、读完 toy 之后

16 个 stage 已经覆盖 VLA 的全部核心概念，是完整的教程主体。之后可做的两件事（都超出「学概念」范畴）：

1. **读真实代码**：用 toy 当地图，逐行读 `alpamayo1_5.py` 的 `sample_trajectories_from_data_with_vlm_rollout`（对照上面「与真实 Alpamayo 的对照」表）。
2. **换真大模型 + 真数据**：把 toy 的合成输入 / 小模型换成 Qwen3-VL + 真实驾驶数据——这是「工程化」，不是「学新概念」。

---

## 附录：扩散理论深读（DDPM vs Flow Matching）

> 这不是 stage（不在 VLA 递进主线里），而是 `stage2` 那个 `FlowMatching` 背后的**理论基础**。想搞懂「扩散到底在干嘛」，看这两个 1D 最小例子（双峰分布 {-1,+1}）：

| 文件 | 流派 | 学什么 | 采样 |
|---|---|---|---|
| `ddpm_1d.py` | DDPM（stable diffusion 同款） | 噪声 ε | 反解 x_0 再重加噪，T 步 |
| `flow_matching_1d.py` | Flow Matching（toy 用的） | 速度 v = x1 - x0 | 欧拉积分，50 步 |

核心对照：
- **路径**：DDPM 用 `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`（有噪声调度）；FM 用 `x_t = (1-t)·x_0 + t·x_1`（直线，无调度）。
- **预测对象**：DDPM 预测噪声 ε；FM 预测速度 v。
- **采样**：DDPM 逐步去噪（500 步）；FM 欧拉积分（50 步，更快更简单）。
- **符号约定注意（易混）**：FM 里 `x_0=噪声, x_1=数据`（下标是时间 t）；DDPM 里 `x_0=数据, x_T=噪声`（下标是加噪步数）。两者相反！

stable diffusion = DDPM（预测 ε）+ 文本 cross-attention（这个 cross-attention 就是 toy 里 Expert 干的事）。
