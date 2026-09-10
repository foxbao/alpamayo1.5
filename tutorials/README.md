# MiniVLA 学习笔记

从零手写一个迷你 VLA（Vision-Language-Action），理解 Alpamayo 1.5 的核心机制。

> ## ⚠️ 定位说明（重要）
>
> **本教程是 Alpamayo 1.5 的「概念化 MiniVLA」，不是一个可复现真实模型的实现。**
> 每个 stage 用最小代码解释**一个**机制；最后（见 §六）再把 toy 模块映射回真实文件。
>
> 真实模型的规模是 toy 的**几个数量级**：真实 VLM 是 8B 参数、15 万词表、30+ 层；
> toy 是几万参数、十几个词、2 层。两者**结构对应，规模不可比**。

> 适合：熟悉 3D 检测 / BEV、有一定 LLM 基础、想系统理解 VLA 的读者。
> 每个 stage 是一个可独立运行的 `.py`，从框架到训练逐步递进。
> toy 代码**只支持 CPU、固定 batch 与固定 shape**，刻意不做工程化。
>
> **另见**：**训练教程**在 `alpamayo-recipes/tutorials/`（讲权重是怎么训出来的）。

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
| 14 | `stage14_complete.py` | 完整输入 + CoC | 历史 + 图片一起进 CosmosReason（toy 里最完整的输入侧；仍与真实有差距，见 §六） |
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

## 六、Toy → Real 对照（与真实 Alpamayo 的差距）

> 本节是本教程的「免责声明 + 地图」：toy 是**教学近似**，这里逐项列出它和真实 release 的差距。

| Toy 模块 | 真实模块 | 真实文件 |
|---|---|---|
| ActionSpace | `UnicycleAccelCurvatureActionSpace` | `action_space/unicycle_accel_curvature.py` |
| FlowMatching | `FlowMatching` | `diffusion/flow_matching.py` |
| ActionInProj/OutProj | `PerWaypointActionInProjV2` / out_proj | `models/action_in_proj.py` |
| Expert | `self.expert`（Qwen3 text transformer，无 embed_tokens） | `models/alpamayo1_5.py` |
| ViT / CosmosReason（因果 transformer，自回归生成 CoC） | `self.vlm`（release 用 Cosmos-Reason2：ViT + LLM，看图自回归生成 CoC） | `models/base_model.py` |
| MiniVLA.sample | `sample_trajectories_from_data_with_vlm_rollout` | `models/alpamayo1_5.py:218` |

### 逐项差距清单

真实值来自 `Alpamayo-1.5-10B/config.json` 与对应源码。

| 方面 | toy | 真实 release | 性质 |
|---|---|---|---|
| VLM | 2~3 层、hidden 64、**随机初始化** | **Cosmos-Reason2-8B**（30+ 层、预训练） | 规模 |
| 词表 | 7~15 个词 | ~15 万（BPE） | 规模 |
| 历史轨迹 | 8~16 步序列 | **48 个 token**（16 位姿 × 3 维） | 表示 |
| 未来轨迹 | 64 waypoints（动作空间 `(64,2)`） | `tokens_per_future_traj=128`、`traj_vocab_size=4000` | 表示 |
| Expert hidden | 64 | **2048**（`expert_cfg.hidden_size`） | 规模 |
| **条件机制** | **显式 cross-attention** `attn(x, cond, cond)` | **KV-cache prefix attention**（VLM 的 KV cache 直接接进 expert 的 `past_key_values`） | ⚠️ **结构差异** |
| 条件长度 | 2~3 个 token | 数十~上百（CoC 文本 + 历史 + 路由） | 规模 |
| 位置编码 | 可学习 `pos_embed` | **RoPE**（旋转位置编码，Qwen 系） | 实现差异 |
| 动作积分 | 固定 `v0` + 欧拉 + Python 循环 | 从历史**估计 v0** + **梯形积分** + `cumsum` 向量化 + **输出旋转矩阵** | 精度/工程 |
| **CFG** | 可学习的「空条件」embedding | **移除导航文本段**（route removal）构造无条件输入 | ⚠️ **结构差异** |
| 图像输入 | 16×16 合成灰度图 | 1920×1080 RGB 多相机，`min/max_pixels` 约束 | 规模 |
| 推理引擎 | 纯 PyTorch 循环 | HuggingFace + DeepSpeed/Hydra | 工程 |

**哪些差距「不影响学概念」**：规模类（层数、hidden、词表）——结构一样，放大即可。
**哪些是「结构性的，要知道不一样」**：上面标 ⚠️ 的两条（条件机制、CFG）。

### 关键的两个「结构差异」（务必知道）

**① 条件机制**：toy 用显式的第二个 attention（`cross_attn(query=动作, key/value=条件)`）；
真实代码是把 VLM 的 **KV cache 对象**直接传给 expert 当 `past_key_values`（`alpamayo1_5.py:304,349`）。

**实测（`real3_kv_cache.py`，CPU）**：

| 对比 | 结果 |
|---|---|
| 输出数值 | **完全相同**（最大差异 `0.00e+00`） |
| 条件的 K/V 投影次数（10 步扩散） | toy 10 次 ↔ 真实 **1 次** |
| 只省投影，实测加速 | 仅 **1.7 倍**（attention 本身仍要 O(L)） |
| **真正的大头**：每步过 transformer 的 token 数 | 前缀+动作（3136）↔ **只有动作（64）** → **34 倍** |

**关键结论**：KV cache 省的不是「条件的投影」，而是**整个前缀的前向计算**——
每步的计算量从 3136 tokens 降到 64 tokens。toy 的写法数学上没错，但在真实规模下会慢 30~50 倍。

**② CFG 的无条件分支**：toy 学了一个「空条件 embedding」来表示「无指令」；
真实代码是**从输入序列里删掉 `<|route_start|>...<|route_end|>` 那一段**（`nav_utils.remove_nav_text`），
再跑一遍 VLM 得到无条件 KV cache（`alpamayo1_5.py:519-575`）。**机制相同，构造方式不同。**

**关于「一个模型 vs 多个模块」**：真实代码里只有一个大模型 `self.vlm`，它**内部**包含三部分，对应 toy 的三个模块。
（release 配置里是 **Cosmos-Reason2-8B**；代码默认值是 `Qwen/Qwen3-VL-8B-Instruct`，两者接口相同。）

| Cosmos-Reason2 内部 | 干什么 | 对应 toy |
|---|---|---|
| Vision Encoder（ViT） | 图 → visual tokens | `build_vit`（stage9） |
| tokenizer + embed_tokens | 词 → 词向量 | `nn.Embedding`（即「Text Encoder」） |
| LLM 的 transformer 层 | 自回归生成 CoC | `CausalBlock` 堆（即「Cosmos Reason Backbone」） |

所以「Text Encoder」和「Cosmos Reason Backbone」**不是两个模型**，而是 Cosmos-Reason2 这一个模型内部的两部分；真正的第二个模型是 `self.expert`（Trajectory Decoder / 去噪器）。

### 实测对照（`real1_input_trace.py`）

上面是读代码总结的；下面是**真实跑一遍**测出来的（纯 CPU，不需加载 10B 模型）：

| 项 | toy（stage13） | 真实（实测） |
|---|---|---|
| 序列长度 | 17 视觉 + 4 文本 | **3086**（2880 视觉 + 约 206 文本） |
| 图片占比 | ~0% | **93%** |
| 每相机帧数 | 1 | **4**（用文字标签 `frame 0..3` 标注） |
| 历史轨迹 | 独立 encoder 输出 | **48 个 `<\|traj_history\|>` 占位符**（位置 3013~3060），待替换 |
| 生成起点 | `<bos>` | `<\|cot_start\|>` 在**序列末尾**（位置 3085） |
| 相机同步 | — | **不同步**：各相机曝光时刻差 ~27ms |
| 时间戳 | 不用 | `relative_timestamps` 算了但**未传给模型** |
| 外参 | 不用 | 不用（靠相机名文字标签 + ego 历史） |

**三个最重要的实测结论**：

1. **序列 93% 是图片**——多相机 × 4 帧把序列几乎塞满视觉 token，这是真实推理吃显存的主因
2. **时间信息只靠「`frame N` 标签 + 序列顺序」传递**，真实时间戳没进模型
3. **外参完全不用**——相机几何是「隐式」学进权重的（详见 §七）

跑法：`python tutorials/real1_input_trace.py`（纯 CPU，约 1 分钟）

**`real2_inference_trace.py`（后半段：tokens → 轨迹，需 GPU）实测**：

| 步骤 | 实测结果 |
|---|---|
| 模型规模 | **11.08 B** 参数 |
| 模型构成 | `vlm = Qwen3VLForConditionalGeneration`（印证 §六 的架构判断）<br>`expert = Qwen3VLTextModel`、`action_space = UnicycleAccelCurvatureActionSpace`、`diffusion = FlowMatching` |
| 历史 token 化 | 位置 3013~3060 的 48 个占位符 → 真实 token id（如 `154669, 155176, ...`） |
| CoC 生成后 | `<\|traj_future_start\|>` 的后一位 = **3103**（diffusion token 从这里开始） |
| 扩散采样 | `batch_size=1, n_steps=10` → 动作 `(1, 64, 2)` |
| 最终输出 | `pred_xyz (1, 1, 1, 64, 3)`、`pred_rot (1, 1, 1, 64, 3, 3)` |

跑法：`CUDA_VISIBLE_DEVICES=1 python tutorials/real2_inference_trace.py`（需 GPU，避开被占用的 GPU 0）

**`real3_kv_cache.py` / `real4_traj_roundtrip.py` —— 两个动手实验**

`real3`：验证 §六 的「结构差异①」（KV cache）——输出**数值完全相同**，但每步过 transformer 的
token 数从 3136 降到 64 → **34 倍**加速。KV cache 省的不是「条件的投影」，是**整个前缀的前向**。

`real4`：轨迹几何 round-trip（用**真实**动作空间 + tokenizer）：

| | 误差 |
|---|---|
| 连续 round-trip（轨迹→动作→轨迹） | mean **0.0344 m** |
| 量化 round-trip（轨迹→token→轨迹） | mean **0.0360 m** |
| **量化额外损失** | **+5%**（0.0016 m） |

**两个结论**：
1. **量化几乎无损**——分辨率达 0.0045 m/s²（加速度）、0.00017 1/m（曲率）。这解释了为什么
   Alpamayo 敢「训练用离散 token、推理用连续 diffusion」：**两者表示同一件事，精度差 5%**。
2. 主要误差来自**连续 round-trip 本身**（带正则的最小二乘反解，本就不是精确逆），不是量化。

> 顺带解开了 `tokens_per_future_traj = 128` 的谜：**128 = 64 waypoints × 2（加速度, 曲率）**。
> 历史则是 `48 = 16 位姿 × 3 (xyz)`。

**`real5_eval_metrics.py` —— 评估指标（别只看 minADE）**

跑法：`CUDA_VISIBLE_DEVICES=1 N_SAMPLES=2 python tutorials/real5_eval_metrics.py`
（24GB 卡最多 2 条采样；4 条会 OOM）

实测（同一条 clip，2 条采样）：

| 类别 | 指标 | 结果 |
|---|---|---|
| **精度** | ADE | min **0.225** / mean 0.616 / max 1.006 m |
| | FDE | min 0.708 / mean **3.011** / max 5.314 m |
| | 航向误差 | mean **0.46°** |
| **舒适性** | 加速度 max | 模型 **1.423** ↔ 人类 0.524 m/s² |
| | 曲率 / jerk | 与真值同量级 |
| **多样性** | 终点两两距离 | 3.011 m |

**三个关键结论**：

1. **`minADE` 会骗人**——它是「K 条里挑最好」，min 0.225 vs mean 0.616（差 3 倍）、max 1.006（差 4.5 倍）。
   真实部署要看**分布**，不能只看 min。
2. **FDE 比 ADE 差 5 倍**（3.0 vs 0.6 m）——误差沿轨迹**累积**，前段的小误差会放大到终点。
3. **⚠️ 模型比人类开得「猛」**：预测的加速度峰值是真值的 **2.7 倍**（1.423 vs 0.524 m/s²）。
   位置误差看着小，但**驾驶风格明显不同**——只看 minADE 完全发现不了。这正是
   `alpamayo1_x_rl` 里 `comfort_reward` 要惩罚的东西。

**`real6_ablation.py` —— 输入消融（模型到底在看什么？）**

跑法：`CUDA_VISIBLE_DEVICES=1 python tutorials/real6_ablation.py`（每次推理约 40s）

固定同一随机种子，每次只改**一个**输入维度：

| 消融 | 终点距离 | 平均轨迹距离 | CoC |
|---|---|---|---|
| **左右相机标签互换** | **0.01 m** | 0.01 m | 相同 |
| 帧序倒转 | 1.49 m | 0.32 m | 相同 |
| 只用 2 个相机 | 4.90 m | 1.06 m | **不同** |
| **历史轨迹置零** | **50.56 m** | 25.76 m | **不同** |

（baseline 终点 52.84 m，故 50.56 m ≈ 整条轨迹塌掉）

**影响力排序：历史轨迹 ≫ 相机图像 > 时序顺序 ≈ 相机文字标签**

三个结论：

1. **历史轨迹是绝对主力**——置零后轨迹从 52.8m 塌到 2.3m（模型不知道当前速度就不预测运动）
2. **相机文字标签几乎不起作用**（互换只差 0.01m）——模型靠**图像内容**而非标签判断相机身份
3. **CoC 对轻微扰动很稳定**——5 个变体里 4 个 CoC 一字不差，只有严重破坏输入才变。
   这印证了「**CoC 是条件信号，不等于忠实解释**」

> ⚠️ 以上是 **n=1**（一条 clip、一次采样）的观察，要下真结论需跑几十条 clip 统计。

---

## 七、VLA vs BEV：为什么不需要显式 3D

三个来自 BEV/UniAD 视角的常见疑问：

1. **多相机怎么知道谁是谁？** 靠**文本标签**——`helper.py` 在每路相机的图前插 `"Front camera: "` 等文本，prompt 是「文本标签 + 图片」交错排列。BEV 用外参硬编码几何，VLA 用语言描述传感器。

2. **外参不需要吗？** 推理时不显式给外参（输入只有图 + ego 历史 + 文本），但外参**隐式学进权重**——相机装固定位置，模型从训练数据学会「前相机画面对应车前方什么位置」。代价：换传感器布局要重训。

3. **不需要猜 3D/投影吗？** 对，无显式深度 / BEV / 投影，模型从 2D 图 + ego 历史**隐式**理解 3D。这是「端到端」vs「模块化 BEV」的分野：BEV 显式可解释、数据量要求低、但对标定敏感；VLA 端到端简单、免标定，但黑盒、需海量数据。

---

## 八、读完 toy 之后

16 个 stage 覆盖了 VLA 的**主要机制**（动作空间、扩散采样、条件化、多模态融合、自回归生成、CFG）。
但请注意：**这是概念化的 MiniVLA，不是对真实模型的复刻**——差距见 §六「Toy → Real 对照」。

之后可做的两件事（都超出「学概念」范畴）：

1. **读真实代码**：用 toy 当地图，逐行读 `alpamayo1_5.py` 的 `sample_trajectories_from_data_with_vlm_rollout`（对照上面「与真实 Alpamayo 的对照」表）。
2. **换真大模型 + 真数据**：把 toy 的合成输入 / 小模型换成 Cosmos-Reason2 + 真实驾驶数据——这是「工程化」，不是「学新概念」。

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
