# MiniVLA 学习笔记

从零手写一个迷你 VLA（Vision-Language-Action），理解 Alpamayo 1.5 的核心机制。

> ## ⚠️ 定位说明（重要）
>
> **本教程是 Alpamayo 1.5 的「概念化 MiniVLA」，不是一个可复现真实模型的实现。**
> 每个 stage 用最小代码解释**一个**机制；最后（见 §六）再把 toy 模块映射回真实文件。
>
> 真实模型的规模是 toy 的**几个数量级**：真实 VLM 是 8B 参数、15 万词表、30+ 层；
> 可训练的 toy（stage6~15）约 16 万～73 万参数、几个词、少量 transformer 层；
> stage0~5 是接口/模块演示。两者有概念对应，但不能视为同一架构的等比例缩小。

> 适合：熟悉 3D 检测 / BEV、有一定 LLM 基础、想系统理解 VLA 的读者。
> 每个 stage 是一个可独立运行的 `.py`；stage0→stage6 是主线，后续章节包含并列和高级专题。
> toy 默认只支持 CPU 和固定序列规格；stage0~5 固定 batch，stage6~15 支持可变 batch。
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

下面以 **stage7 的历史条件分支**为例。stage9 起加入图片，stage13/14 加入文本生成。

```
输入
├─ 历史轨迹序列 (B, H, 2)
└─ （stage7 省略图片，后续 stage 再加入）

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
| 1 | `stage1_action_space.py` | 单轮车运动学 | 动作 = (加速度, 曲率)，积分成轨迹；运动学约束不等于安全/舒适保证 |
| 2 | `stage2_flow_matching.py` | 向量场采样直觉 | 在已知的 `target - x` oracle 收缩场上做欧拉积分；真正的 flow matching 训练见 stage6 |
| 3 | `stage3_projections.py` | Fourier 编码 | 动作/时间编码成 embedding，进 transformer 的「桥」 |
| 4 | `stage4_expert.py` | **条件化的两种做法** | ① cross-attention（通用做法）② **prefix 续写（Alpamayo 真实做法）**——动作 token 接在 VLM 的逐层 K/V 后面，Expert 无 cross-attn |
| 5 | `stage5_assembly.py` | 组装 | 把 1~4 拼成完整 MiniVLA，跑通闭环 |
| 6 | `stage6_training.py` | flow matching 训练 | 预测向量场 v = x1 - x0，离散 condition 控制轨迹 |
| 7 | `stage7_history_condition.py` | 序列编码条件 | condition 从历史轨迹「读」出来，逼近真实 VLM |
| 8 | `stage8_multimodal.py` | 多峰分布（并列分支） | 同一条件采出左转/右转两簇；单输出 MSE 回归只拟合条件均值 |
| 9 | `stage9_vision.py` | 视觉编码（ViT 结构） | 随机初始化的 Transformers ViT 把图片编码成 visual tokens 当 condition |
| 10 | `stage10_text.py` | 文本编码 | 文本 token → embedding → transformer → condition |
| 11 | `stage11_fusion.py` | 多模态融合 | 三路 token concat + 位置编码，再由 Expert 读取（⚠️ 三路在此为冗余，只演示「怎么合」，不证明「为何必须合」） |
| 12 | `stage12_two_cameras.py` | 多相机（文本标签） | 共享 ViT + 每路标签/图片上下文化，再 concat |
| 13 | `stage13_coc.py` | 自回归 CoC 生成 | 因果 transformer **用 KV cache 增量**自回归生成 CoC（**变长，见 EOS 停**）；**Part 2 直接用这份 cache 做 prefix 条件化并对比** |
| 14 | `stage14_complete.py` | 完整输入 + CoC | 历史 + 图片拼成**多模态前缀**再生成 CoC，cache 同样直接复用（toy 里最完整的输入侧；与真实的差距见 §六） |
| 15 | `stage15_cfg.py` | CFG 引导 | 条件丢弃训练 + `v=(1-w)·v_uncond + w·v_cond`，w>1 外推向量场 |

**演进脉络（每个 stage 相对上一个改了什么）：**

| 过渡 | 改了什么 |
|---|---|
| 0→5 | 自包含、逐个搭模块（接口→动作空间→扩散→投影→Expert→组装） |
| 5→6 | FlowMatching 的 batch 参数化（推理固定 batch → 训练可变 batch）；加训练循环 |
| 6→7 | 并列分支：condition 从「离散 Embedding 查表」→「历史序列 → ConditionEncoder」 |
| 6→8 | 并列分支：做「多模态」demo（用固定 condition，κ 从 0.05 放大到 0.2 让两模式分得开） |
| 6→9 | 并列分支：补「V」，condition 从随机初始化的 ViT 结构编码图片而来 |
| 6→10 | 并列分支：补「L」，condition 从「文本 token」来 |
| 9/10→11 | 融合：三路 token concat + 位置编码；玩具输入都泄露相同方向，不能据此证明融合必要性 |
| 9→12 | 并列分支：双相机共享 ViT，每路标签与图片经 transformer 交互后拼接；不含历史/导航 |
| 9→13 | 并列分支：单张图 + 因果 transformer 生成短文本，隐状态当 condition；不继承双相机 |
| 13→14 | 补回历史输入：历史 + 图片融合进 CosmosReason 输入序列，再自回归生成 CoC |
| 6→15 | 并列分支：离散条件 + 条件丢弃训练（50% 空条件）+ CFG；不继承视觉/CoC |

> 运行提示：`common.py` 是 stage4 之后抽出的**共享积木**（ActionSpace / FlowMatching / 投影 / Expert + 常量）；stage6~15 通过 `from common import ...` 复用，运行前需 `common.py` 在同目录。stage0~5 仍是自包含的（造积木阶段）。

在仓库根目录、已安装依赖的 Python 3.12 环境中运行：

```bash
python tutorials/stage0_framework.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tutorials/stage6_training.py
python -m pytest -q tests/test_tutorials.py
```

线程数是小模型 CPU 实验的可选设置；数千步训练不保证秒级完成。
stage0~15 与两个 1D 示例不下载权重；`real1/4` 需要 HF 配置、tokenizer 或 gated 数据，
`real2/5/6` 还需 CUDA、约 22 GB 模型权重和足够显存。`real3` 是纯 CPU 合成实验。

### 建议学习顺序

先按主线运行 **stage0→stage6**，建立模块接口、动作空间、采样循环、条件化和训练目标。
从 stage6 开始，**stage7→stage12 是并列专题**，可按兴趣选择：历史条件、多模态、视觉、文本、融合和多相机，
不要求逐章运行，也不互相继承全部代码。理解 stage9/10 的条件输入后，再看 **stage13→stage14 高级专题**（CoC、
因果 mask、teacher forcing 与完整输入）；**stage15 CFG** 是另一条独立的高级分支（从 stage6 出发），不是 stage14 的必然后继。
最后再选读 **real1→real6**：它们面向真实 release 的输入追踪、推理、缓存、几何 round-trip、评估和消融，可能需要
Hugging Face 权限、数据和 GPU，不是 toy 主线的强制前置。

两个可选实验（不属于主线，默认不纳入回归测试）：
- `tutorials/exp_fourier.py` —— stage3 之后，观察 Fourier 频率数量对拟合高频函数的影响。
- `tutorials/exp_kv_cache.py` —— stage13 之后，对比「每步重算前缀」与「prefill + KV cache」：
  输出相同，计算量随长度平方拉开（长度 4→512 时差距 7→384 倍）。
- `tutorials/exp_prefix_expert.py` —— 用 **Alpamayo 真实的 prefix 方式**实现条件化：
  VLM 产 cache（逐层 K/V）→ Expert 接在后面算。**没有 cross-attention，没有独立的 condition 张量**；
  条件完全通过 cache 传递。跑通并训练到收敛，可和 stage4~15 的 cross-attention 版对照。

---

## 四、关键概念速查

| 概念 | 一句话 |
|---|---|
| **condition** | VLM（或 encoder）输出的隐状态，编码了「图 + 历史 + 推理」，是 Expert 的 cross-attn 的 key/value |
| **向量场** | 给动作空间每个点配一个「去噪方向」，`step_fn(x,t)` 算当前点的方向 |
| **flow matching** | 学一个向量场把噪声「推」向数据；训练目标 v = x1 - x0 |
| **cross-attention** | query=动作 token，key/value=条件 token，让条件塑造动作 |
| **非因果 self-attn** | Expert 的动作 token 互相可见（轨迹是整体，不像 LLM 从左到右） |
| **action space** | 用 (accel, curvature) 积分 xyz；限制运动学形式，不自动保证碰撞安全或 jerk 平滑 |
| **Fourier 编码** | 标量 → 多频率 sin/cos，让 MLP 能表示高频函数（同 NeRF 位置编码） |
| **temperature** | toy 中缩放初始高斯噪声；改变它会改变采样分布，效果需验证，不保证更稳定 |
| **CFG** | `v=(1-w)·v_uncond + w·v_cond`；w=0 为边缘分布，w=1 为条件分布，w>1 外推向量场 |

---

## 五、踩过的坑（实战经验）

1. **小信号与积分误差**：目标曲率 κ=±0.05 相对 σ=1 的噪声很小。向量场 MSE 是跨动作维度和时间的平均，不能直接换算成固定曲率误差或几米的轨迹误差；需分别报告动作误差和积分后的 ADE/FDE。

2. **先对齐训练/采样噪声**：本教程都在标准高斯下训练，默认用 temperature=1.0 采样。推理可调整温度，但不能保证更好，也不意味着必须重训。真实入口的 `temperature=0.6` 控制文本生成，动作噪声温度在 `diffusion_kwargs` 中，是两个参数。

3. **模式切换与运行成本**：训练用 `model.train()`，评估用 `model.eval()` + `torch.no_grad()`；后者不会自动关闭 dropout。stage6~15 约 16 万～73 万参数，CPU 可运行，但速度也取决于 batch、序列长度和线程数。真实模型还包含 Expert，不能仅按 VLM 的 8B 估算显存。

   **本教程刻意只用 CPU**：全部脚本不含任何 `.to(device)`，在哪儿都能跑，读代码时不会被设备管理干扰。
   代价是慢 —— 但慢的原因**不是**没上 GPU：这些张量最大只有 `64×64×64`，单步耗时里
   绝大部分是 Python 调度和 autograd 建图，真正的算数只占微秒级。实测同一训练步：
   CPU 8 线程 65ms / CPU 56 线程（默认）84ms / **4090 也只要 35ms**——只快 2.3 倍，
   因为 GPU 省不掉 Python 那一层。`common.py` 因此显式 `torch.set_num_threads(8)`：
   小张量上默认的 56 线程同步开销大于收益（`time` 里 `user` 是 `real` 的 50 多倍就是这个原因）。

4. **不要把教学观察当成收敛保证**：stage2 的 `target-x` 只是在演示收缩场，并非学得的直线流匹配速度；stage8 的平均曲率不能替代逐轨迹检查。stage13/14 按 EOS 停止做变长生成（推理链长度 6/6/3），训练时短的补齐、pad 位置用 `ignore_index` 跳过；condition 也补齐到定长，并用 `key_padding_mask` 让 Expert 的 cross-attention 忽略 pad（对应真实代码的 `_build_expert_pos_ids_and_attn_mask`）。值得记住：`ignore_index` 只让 pad 不参与 loss，**并不会阻止 pad 进入 transformer**——要真正屏蔽必须靠 attention mask。teacher forcing 与生成前缀仍有分布差异。

   **同一个 pad 问题在 prefix 侧会再出现一次**：变长生成会给已结束的样本补 `<pad>`，
   这些位置也进了 cache、成了 Expert 的前缀，所以 Part 2 还要一个 `cache_pad_mask`
   在 attention 里把它们屏蔽掉。位置不同（条件张量 / 前缀 K/V），道理完全一样。

5. **KV cache：stage13/14 已经是增量的**。`generate` 先 prefill 一次，之后每步只把
   **新 token** 喂进 CosmosReason（O(n)），产出的 cache 直接交给 prefix 版 Expert。
   为什么能这样：因果 mask 让位置 j 的 K/V 只依赖 token 0..j，后面追加多少 token 都不影响它。
   两种写法**输出完全相同**，只是计算量随长度平方拉开（`exp_kv_cache.py` 实测：长度 4→512 时差距 7→384 倍）。
   注意 Part 1（cross-attn 版）**仍然要再跑一次完整 forward** —— 因为 cross-attention 要的是
   hidden 张量而不是 K/V。这个「多出来的第二次 forward」正是两种连接方式的代价差异，
   真实代码里没有它（`alpamayo1_5.py:304` 直接把 `past_key_values` 传给 expert）。

6. **CFG 不等于曲率放大器**：stage15 对左/右目标随机丢弃条件，空条件学习两者的边缘分布，不是直行目标。各 w 共用初始噪声；w>1 不保证曲率单调增大，更不保证比条件采样安全或准确。

---

## 六、Toy → Real 对照（与真实 Alpamayo 的差距）

> 本节是本教程的「免责声明 + 地图」：toy 是**教学近似**，这里逐项列出它和真实 release 的差距。

| Toy 模块 | 真实模块 | 真实文件 |
|---|---|---|
| ActionSpace | `UnicycleAccelCurvatureActionSpace` | `action_space/unicycle_accel_curvature.py` |
| FlowMatching | `FlowMatching` | `diffusion/flow_matching.py` |
| ActionInProj/OutProj | `PerWaypointActionInProjV2` / out_proj | `models/action_in_proj.py` |
| Expert | `self.expert`（Qwen3 text transformer，无 embed_tokens） | `models/alpamayo1_5.py` |
| ViT / CosmosReason（随机初始化的因果 transformer，自回归生成 CoC） | `self.vlm`（release 用 Cosmos-Reason2：ViT + LLM，看图自回归生成 CoC） | `models/base_model.py` |
| MiniVLA.sample | `sample_trajectories_from_data_with_vlm_rollout` | `models/alpamayo1_5.py:218` |

### 逐项差距清单

真实值来自 `Alpamayo-1.5-10B/config.json` 与对应源码。

| 方面 | toy | 真实 release | 性质 |
|---|---|---|---|
| VLM | 2~3 层、hidden 64、**随机初始化** | **Cosmos-Reason2-8B**（30+ 层、预训练） | 规模 |
| 词表 | 2~7 个词（有文本的 stage） | ~15 万（BPE） | 规模 |
| 历史轨迹 | 8~16 步序列 | **48 个 token**（16 位姿 × 3 维） | 表示 |
| 未来轨迹 | 64 waypoints（动作空间 `(64,2)`） | `tokens_per_future_traj=128`、`traj_vocab_size=4000` | 表示 |
| Expert hidden | 64 | **2048**（`expert_cfg.hidden_size`） | 规模 |
| **条件机制** | **显式 cross-attention**：动作一路、条件一路，用 `attn(x, cond, cond)` 连接（stage13/14 的 Part 1）。<br>Part 2 已实现真实做法：动作 token 续写在前缀 K/V 后面 | **prefix 续写**：**真实 Expert 没有任何 cross-attention**——它和 VLM 文本塔**结构完全相同**（`self_attn + MLP`），动作 token 直接**接在 VLM 序列后面**，用 self-attention 看前缀 | ⚠️ **连接拓扑不同**（Part 1 vs Part 2 就是这个对比） |
| 条件来源 | 条件张量 / generate 留下的 cache | VLM `generate()` 留下的 `past_key_values`，**含自己生成的 CoC** | 一致（Part 2） |
| 条件长度 | 2~36 个 token（依 stage 而变） | 完整多模态前缀的逐层 K/V，含视觉，示例超过 3000 个位置 | 规模/表示 |
| 位置编码 | 可学习 `pos_embed` | **RoPE**（旋转位置编码，Qwen 系） | 实现差异 |
| 动作积分 | 固定 `v0` + 欧拉 + Python 循环 | 从历史**估计 v0** + **梯形积分** + `cumsum` 向量化 + **输出旋转矩阵** | 精度/工程 |
| **CFG** | 可学习的「空条件」embedding | **移除导航文本段**（route removal）构造无条件输入 | ⚠️ **结构差异** |
| 图像输入 | 16×16 合成灰度图 | 1920×1080 RGB 多相机，`min/max_pixels` 约束 | 规模 |
| 推理引擎 | 纯 PyTorch 循环 | PyTorch + Transformers；Hydra 实例化配置，本仓库不依赖 DeepSpeed | 工程 |

**规模差异**不妨碍理解模块接口，但放大参数并不足以复现预训练、数据分布或实际驾驶能力。
**哪些是「结构性的，要知道不一样」**：上面标 ⚠️ 的两条（条件机制、CFG）。

### 关键的两个「结构差异」（务必知道）

**① 条件机制**：toy 用显式的第二个 attention（`cross_attn(query=动作, key/value=条件)`）；
真实代码是把 VLM 的 **KV cache 对象**直接传给 expert 当 `past_key_values`（`alpamayo1_5.py:304,349`）。
这是不同的 attention 结构：真实动作 query 同时读取前缀和动作 K/V；toy 分两次 attention 计算。
两者都能条件化，但并非数学等价。

**实测（`real3_kv_cache.py`，CPU）**：

| 对比 | 结果 |
|---|---|
| 输出数值 | 同一个 toy cross-attention，缓存与重算 K/V 的输出相同（浮点误差范围内） |
| 条件的 K/V 投影次数（10 步扩散） | 重算版 10 次 ↔ 缓存版 1 次 |
| 只省投影的耗时 | 脚本在无梯度、预热后报告三次中位数；不是 release 模型测速 |
| **计算量示意**：每步处理的 token 数 | 前缀+动作（3136）↔ **只有动作（64）** → token 数约 49 倍；实际端到端加速取决于缓存和 kernel |

**关键结论**：KV cache 可以避免每个扩散步重复计算前缀；上面的 token 数只是直观示意，
不应直接当成真实模型的加速倍数。真实收益还取决于 attention kernel、缓存布局和硬件。
后半段的“只跑动作”根本没有读前缀 K/V，所以也不能作为缓存分支的精确成本或严格上界。

**①-b 两种「条件化」的设计哲学（比上面那条更根本）**

```
toy（两路 · cross-attention）:
    动作 token ──► [Expert] ──┐
                              ├─ cross-attn 连接两路
    条件 (B,L,64) ────────────┘

真实（一路 · prefix 续写）:
    VLM 的 KV cache（前缀）┐
                           ├─► 拼成【一条序列】──► [Expert] ──► 输出
    动作 token ────────────┘
```

实测（`Qwen3VLTextModel` 的第 0 层）：真实 Expert 的子模块只有
`self_attn + mlp + 2×RMSNorm`——**和 VLM 文本塔的第 0 层一模一样**，没有 cross-attention。

所以差异不是「换了一种 attention」，而是「**完全不同的连接方式**」：
- toy：两条数据流，靠 cross-attention 沟通（这也是 Stable Diffusion 文本条件的做法）
- 真实：一条数据流，动作 token「续写」在前缀后面，靠 self-attention 沟通

**两者都是合法的 VLA 设计，但 toy 用的不是 Alpamayo 的设计。** 要让 toy 与真实一致，需要：
① 层数对上；② **去掉 ExpertBlock 的 cross-attention**，改成与 `CacheBlock` 同类型；③ hidden/heads 对齐。

**stage13/14 的 Part 2 就是这个对照实验**：同一个 backbone 产出的 cache，分别用
cross-attn（传 hidden 张量）和 prefix（传逐层 K/V）接给 Expert —— 两种方式都能把条件
传给动作，且 Part 2 里 `step_fn` 的签名上**根本没有 `condition` 参数**。

**② CFG 的无条件分支**：toy 学了一个「空条件 embedding」来表示「无指令」；
真实代码是**从输入序列里删掉 `<|route_start|>...<|route_end|>` 那一段**（`nav_utils.remove_nav_text`），
再跑一遍 VLM 得到 KV cache（`alpamayo1_5.py:519-575`）。这个分支仍有视觉和历史，
只是相对于导航文本“无条件”，并不是移除所有条件。

**关于「一个模型 vs 多个模块」**：真实代码里只有一个大模型 `self.vlm`，它**内部**包含三部分，对应 toy 的三个模块。
（release 配置里是 **Cosmos-Reason2-8B**；代码默认值是 `Qwen/Qwen3-VL-8B-Instruct`，两者接口相同。）

| Cosmos-Reason2 内部 | 干什么 | 对应 toy |
|---|---|---|
| Vision Encoder（ViT） | 图 → visual tokens | `build_vit`（stage9） |
| tokenizer + embed_tokens | 词 → 词向量 | `nn.Embedding`（即「Text Encoder」） |
| LLM 的 transformer 层 | 自回归生成 CoC，并留下 `past_key_values` 给 Expert | `CosmosReason`（`CacheBlock` 堆）；生成出的 cache 就是 Expert 的前缀 |

所以「Text Encoder」和「Cosmos Reason Backbone」**不是两个模型**，而是 Cosmos-Reason2 这一个模型内部的两部分；真正的第二个模型是 `self.expert`（Trajectory Decoder / 去噪器）。

### 实测对照（`real1_input_trace.py`）

以下 real1~6 表格是先前单条 clip 的运行记录，不是本次修正后的完整复跑结果。
参数量、token 数和指标应以当前脚本输出为准；real1 纯 CPU，但仍需处理器与 gated 数据访问。

| 项 | toy（stage13） | 真实（实测） |
|---|---|---|
| 序列长度 | 起始 17 视觉 + 1 BOS，固定生成后共 21；构建条件时用 20 | **3086**（2880 视觉 + 约 206 文本） |
| 图片占比 | 输入起始约 94%，条件重算时 85% | **93%** |
| 每相机帧数 | 1 | **4**（用文字标签 `frame 0..3` 标注） |
| 历史轨迹 | stage13 无历史；stage14 加连续历史编码 | **48 个 `<\|traj_history\|>` 占位符**（位置 3013~3060），待替换 |
| 生成起点 | `<bos>` | `<\|cot_start\|>` 在**序列末尾**（位置 3085） |
| 相机同步 | — | **不同步**：各相机曝光时刻差 ~27ms |
| 时间戳 | 不用 | `relative_timestamps` 算了但**未传给模型** |
| 外参 | 不用 | 不用（靠相机名文字标签 + ego 历史） |

**三个最重要的实测结论**：

1. **这条输入的序列 93% 是图片**，视觉 token 显著影响 KV cache 和计算量；总显存还包括模型权重、采样数及中间张量，不能只归因于图片。
2. **时间信息只靠「`frame N` 标签 + 序列顺序」传递**，真实时间戳没进模型
3. **此推理接口不显式输入相机外参**，但这不能证明模型学到了准确几何，也不表示换相机布局无需验证（见 §七）。

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

跑法：`python tutorials/real2_inference_trace.py`（按实际机器设置 `CUDA_VISIBLE_DEVICES`，不要默认有 GPU 1）

**`real3_kv_cache.py` / `real4_traj_roundtrip.py` —— 两个动手实验**

`real3`：演示缓存原理，同一 cross-attention 的缓存/重算输出相同，并展示两件事：
条件 K/V 从每个扩散步投影一次变成只投影一次；长前缀的 transformer 前向更昂贵。后半部分是
序列长度成本的示意，真实端到端加速仍取决于缓存实现、kernel 和硬件。

`real4`：轨迹几何 round-trip（用**真实**动作空间 + tokenizer）：

| | 误差 |
|---|---|
| 连续 round-trip（轨迹→动作→轨迹） | mean **0.0344 m** |
| 量化 round-trip（轨迹→token→轨迹） | mean **0.0360 m** |
| **相对真值的平均误差变化** | 约 **+5%**（0.0016 m），不等于两种重建的距离 |

**两个结论**：
1. 这条 clip 的量化 round-trip 误差只略有增加，但不代表所有轨迹的量化损失为 5%。
   脚本现在额外报告两种重建之间的距离；误差之差可能受到抵消影响。
2. 连续 round-trip 包含带正则的反解和积分误差；不能仅凭这个实验推断训练流程，
   也不能把量化误差与模型预测精度等同。

> 顺带解开了 `tokens_per_future_traj = 128` 的谜：**128 = 64 waypoints × 2（加速度, 曲率）**。
> 历史则是 `48 = 16 位姿 × 3 (xyz)`。

**`real5_eval_metrics.py` —— 评估指标（别只看 minADE）**

跑法：`N_SAMPLES=1 python tutorials/real5_eval_metrics.py`；需要多样性统计时增加到 2 或更多。
24GB 卡的容量取决于可用显存、输入长度和 attention 后端；历史运行中 4 条曾 OOM，不是通用上限。

下面的数值是一次历史运行记录（同一条 clip、2 条采样），仅用于说明输出格式和量级；重新运行会因采样和环境不同而变化：

| 类别 | 指标 | 结果 |
|---|---|---|
| **精度** | ADE | min **0.225** / mean 0.616 / max 1.006 m |
| | FDE | min 0.708 / mean **3.011** / max 5.314 m |
| | 航向误差 | mean **0.46°** |
| **运动学代理** | 最低 ADE 样本的反解加速度 max | 模型 **1.423** ↔ 真值 0.524 m/s² |
| | 曲率 / jerk | 与真值同量级 |
| **多样性** | 终点两两距离 | 以脚本实际输出为准 |

**三个关键结论**：

1. **`minADE` 会骗人**——它是「K 条里挑最好」，min 0.225 vs mean 0.616（差 3 倍）、max 1.006（差 4.5 倍）。
   真实部署要看**分布**，不能只看 min。
2. 此例的 mean FDE 约为 mean ADE 的 5 倍，说明终点误差高于全程平均；不证明每一步误差单调增长，也不能仅凭两项统计判断误差来源。
3. **舒适性必须与位置误差一起看**：运行记录里的预测加速度峰值高于真值（1.423 vs 0.524 m/s²）。
   这说明位置误差较小也不代表驾驶风格一致；应把加速度、曲率和 jerk 与 ADE/FDE 一起报告。
   当前脚本输出每条样本的运动学统计。反解会平滑动作、裁剪曲率，不能据此验证原始控制量
   是否超限；动作空间边界也不是舒适性阈值。

**`real6_ablation.py` —— 输入消融（模型到底在看什么？）**

跑法：`python tutorials/real6_ablation.py`（需要多次 GPU 推理，耗时依环境而定）

历史实验只在整段推理开头固定种子；文本生成长度变化后，轨迹噪声可能不同。
修正后的脚本在 diffusion 入口再次设置种子，下表旧数值不能当成严格配对消融结果：

| 消融 | 终点距离 | 平均轨迹距离 | CoC |
|---|---|---|---|
| 左右相机标签互换 | 0.02 m | 0.01 m | 相同 |
| 帧序倒转 | 2.35 m | 0.71 m | 相同 |
| 只用 2 个相机 | 3.24 m | 1.17 m | **不同** |
| **【隔离】只改 VLM 历史条件** | **11.05 m** | 4.08 m | **不同** |
| **【混淆】历史全置零**（连 v0 一起） | **44.26 m** | 24.00 m | **不同** |

**关键：把「VLM 条件」和「积分初速度 v0」分离开**（脚本里的【隔离】行：只把送进
prompt 的历史 token 置零，而 `ego_history_xyz/rot` 保持真实，使 `estimate_t0_states`
算出的 v0 不受影响）：

```
历史全置零的总影响  44.26 m
  ├─ VLM 条件部分    11.05 m   (25%)   ← VLM 对历史的使用：中等
  └─ v0 部分        ~33 m      (75%)   ← 主导，但这是【积分机制】的效应
```

**所以不能说「历史是 VLM 的绝对主力」**——大部分差异来自 v0 被置零（车从速度 0
开始积分，自然几乎不动），而不是 VLM 忽略了历史。

不能据此给出普遍的“模态影响力排序”：

1. 历史既进入 VLM，也用于 `estimate_t0_states` 估计积分初速度。置零同时改变两条路径，
   不是单独测试 VLM 对历史的依赖；还属于偏离训练分布的干预。
2. 一条 clip 上标签互换影响小，不代表标签普遍无用，也不能证明模型只靠图像辨认相机。
3. CoC 是否相同只是文本观察，不能证明其推理忠实性；需更多样本和有针对性的因果实验。

> 这些仍是 **n=1** 的探索性观察。需要多 clip、多种子、配对噪声和误差统计才能推广。

---

## 七、VLA vs BEV：为什么不需要显式 3D

三个来自 BEV/UniAD 视角的常见疑问：

1. **多相机怎么区分？** `helper.py` 在各相机图前插文字标签，并保留序列顺序；这是身份线索，不等于精确几何。真实 VLM 对完整序列做上下文化；stage12 也先让本路图片 token 读取标签，不能只靠裸 concat 的相邻关系。

2. **外参不需要吗？** 本仓库的推理接口不显式接收外参。模型可能从训练数据中学习相机布局相关性，但是否能泛化到新布局需要评估，不能直接断言免标定或一定要重训。

3. **不需要显式 BEV 吗？** 此推理链路没有单独的深度/BEV 模块，但输出轨迹仍有三维坐标约定。
   缺少显式模块不能证明模型准确理解了 3D，也不能据此比较数据效率、可解释性或安全性。

---

## 八、读完 toy 之后

16 个 stage 覆盖了 VLA 的**主要机制**（动作空间、扩散采样、条件化、多模态融合、自回归生成、CFG）。
但请注意：**这是概念化的 MiniVLA，不是对真实模型的复刻**——差距见 §六「Toy → Real 对照」。

之后可做的两件事（都超出「学概念」范畴）：

1. **读真实代码**：用 toy 当地图，逐行读 `alpamayo1_5.py` 的 `sample_trajectories_from_data_with_vlm_rollout`（对照上面「与真实 Alpamayo 的对照」表）。
2. **对接真实模型与数据**：还需处理坐标、时间同步、tokenization、预训练、训练目标与评估协议，不是简单替换模块即可完成。

---

## 附录：扩散理论深读（DDPM vs Flow Matching）

> 这不是 stage（不在 VLA 递进主线里）。`stage2` 只是已知 oracle 向量场上的采样直觉演示；想系统理解可训练的 DDPM/Flow Matching，建议看下面两个 1D 最小例子（双峰分布 {-1,+1}）：

| 文件 | 流派 | 学什么 | 采样 |
|---|---|---|---|
| `ddpm_1d.py` | DDPM（随机后验采样） | 噪声 ε | 反解 x_0，再按后验均值/方差采样，T 步 |
| `flow_matching_1d.py` | Flow Matching（toy 用的） | 速度 v = x1 - x0 | 欧拉积分，50 步 |

核心对照：
- **路径**：DDPM 用 `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`（有噪声调度）；FM 用 `x_t = (1-t)·x_0 + t·x_1`（直线，无调度）。
- **预测对象**：DDPM 预测噪声 ε；FM 预测速度 v。
- **采样**：本例 DDPM 注入后验噪声（500 步），FM 欧拉积分（50 步）；步数与质量的权衡取决于模型和求解器，不能推广为固定速度比。
- **符号约定注意（易混）**：FM 里 `x_0=噪声, x_1=数据`（下标是时间 t）；DDPM 里 `x_0=数据, x_T=噪声`（下标是加噪步数）。两者相反！

这两个一维示例只说明训练目标和采样路径，不代表任何完整的图像生成系统。
