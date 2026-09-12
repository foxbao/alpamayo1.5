# MiniVLA 教程复习题（stage0 ~ stage15）

> **用法**：先自己答，再翻到文末对答案。每题标了「考点」，答不上来就回去看那一行代码。
> 答案在意不在字——说出关键点即可。

---

## Stage 0 — 框架

**0-1** `MiniVLA.sample` 只有三行核心逻辑，是哪三步？为什么必须是这个顺序？
<sub>考点：条件 → 扩散 → 转轨迹，以及「条件必须先算」</sub>

**0-2** `vlm_kv` 是 `(B, 32, 64)`、`emb` 是 `(B, 64, 64)`、`traj` 是 `(B, 64, 3)`。
这三个形状里的 `64` 有几个是同一个含义？分别是什么？
<sub>考点：中间维=时间轴，最后维=特征轴，两个 64 巧合</sub>

**0-3** 如果把 `model.vlm_kv = vlm_kv` 这一行删掉（不传条件），代码还能跑吗？输出会有什么变化？
<sub>考点：condition 的作用是「塑造」而非「必需」</sub>

---

## Stage 1 — 动作空间

**1-1** 为什么用 `(加速度, 曲率)` 而不是直接输出 64 个 xyz？至少说两个理由。
<sub>考点：低维、运动学一致；但不自动保证碰撞安全或 jerk 平滑</sub>

**1-2** 输入「全零动作」，积分出来的轨迹是什么样？为什么？
<sub>考点：v 恒定、θ 恒定 → 直线</sub>

**1-3** 正曲率对应左转还是右转？这取决于什么约定？
<sub>考点：ego 坐标系（车头 +x，y 在左），θ 累加方向</sub>

**1-4** toy 里 `v0=5.0` 是写死的；真实代码里这个初始速度从哪来？
<sub>考点：`estimate_t0_states` 从历史轨迹估计</sub>

---

## Stage 2 — 向量场采样直觉

**2-1** 采样循环 `x = x + dt * step_fn(x, t)` 里，`x` 和 `t` 分别是什么？
<sub>考点：x=当前噪声动作，t=去噪进度</sub>

**2-2** 跑出来 std 从 `1.08` 降到 `0.325`，说明了什么？
<sub>考点：本章使用人为提供的 oracle 收缩场；只演示采样循环，不是训练得到的 FM 速度</sub>

**2-3** `temperature` 参数控制什么？调小会有什么效果和代价？
<sub>考点：缩放初始噪声；小 → 稳定但多样性低</sub>

**2-4** 如果 `step_fn` 返回 `x - target` 而目标又要求 `target - x`，循环会怎样？
<sub>考点：方向反了 → 越走越远，std 变大</sub>

---

## Stage 3 — 投影与 Fourier 编码

**3-1** 为什么要先把 2 维的噪声动作「撑」到 64 维？直接用 2 维不行吗？
<sub>考点：transformer 需要足够丰富的特征才能做 attention</sub>

**3-2** Fourier 编码把标量变成 `[sin(2πf₁x), cos(2πf₁x), ...]`，为什么不直接喂原始标量？
<sub>考点：MLP 难表示高频函数，多频率基函数帮它</sub>

**3-3** 时间步 `t` 是怎么进入网络的？为什么必须进去？
<sub>考点：也做 Fourier 编码后 concat；t=0 和 t=0.9 必须可区分</sub>

**3-4** `(B,64,2)` → `(B,64,64)` → `(B,64,2)` 中间的 transformer 是谁？在 toy 的哪一步插进去？
<sub>考点：是 Expert，插在 InProj 和 OutProj 之间</sub>

---

## Stage 4 — 条件化（两种做法）

> stage4 现在并排教两种「把条件交给动作」的方式：
> **① cross-attention**（通用做法）· **② prefix 续写**（Alpamayo 的真实做法）

**4-1** ① cross-attention 版里，`self_attn(x, x, x)` 和 `cross_attn(x, cond, cond)` 的
**query / key / value 分别是谁**？
<sub>考点：self→(x,x,x)；cross→query=动作, key/value=条件</sub>

**4-2** ① 版里，为什么 Expert 的 self-attention 是**非因果**的？和 LLM 的因果注意力差在哪？
<sub>考点：轨迹是整体，waypoint 之间要互相可见；LLM 是从左到右生成</sub>

**4-3** ② prefix 版里，动作 token 是怎么「看到」条件的？**和 ① 的本质不同在哪？**
<sub>考点：动作 token 接在 VLM 的逐层 K/V 后面，靠 self-attention 看前缀；
① 是另开一路用 cross-attention 连接</sub>

**4-4** **真实 Alpamayo 用的是哪一种？** 为什么真实能让 Expert 直接复用 VLM 的
**KV cache**，而 ① 版做不到？
<sub>考点：真实用 ②（实测 Expert 无 cross-attn）；能复用是因为 Expert 与 VLM 文本塔
**结构相同**（同层数、同 KV 头数、同 head_dim），cache 能直接塞进 past_key_values</sub>

**4-5** 怎么用**实验**证明「条件真的被读进去了」？两种做法各怎么验？
<sub>考点：改条件 → 看输出是否变。① 比较「随机条件 vs 全 0 条件」；
② 比较「换一组 cache vs 原 cache」（因为我们跑过：0.1002 和 0.0281）</sub>

**4-6** ① 和 ② 是「**进阶关系**」还是「**对等关系**」？那 toy 主线（stage4~12）为什么选 ①？
<sub>考点：**对等** —— 两种连接拓扑，不存在「①学会了才轮到②」。
toy 选 ① 是有理由的：条件是显式的 `(B,L,H)` 张量，stage7~12 能直接
`print(condition.shape)` 看清「历史 8 步」「图片 17 个 patch」怎么拼的；
② 的条件埋在 K/V 里，只能看到 `K(1,4,32,64)`，看不出里面装了什么。
⇒ 注意别把「教学上方便」误读成「①更初级」</sub>

**4-7** 推理时 **① 比 ② 多付了什么代价**？为什么？这解释了真实代码里的哪一行？
<sub>考点：① 的 cross-attn 要的是 **hidden 张量**，所以生成完还得**再跑一次完整
forward** 才拿得到条件；② 要的是 **K/V**，`generate()` 时本来就会产生，**白拿**。
⇒ 这正是真实代码能一句 `expert(..., past_key_values=prompt_cache)` 的原因
（stage13/14 的 Part 1 vs Part 2 把这条代价差直接跑出来了）</sub>

**4-8** ② 里 `PrefixVLM` 和 `PrefixExpert` 用的是**同一种 block 吗**？为什么必须这样？
<sub>考点：都是 `CacheBlock`、**同层数**（这里都是 2 层）。不同就接不上——
cache 是「每一层一份 K/V」，层数或结构不一致，Expert 拿到的东西维度/语义都不是一套。
⇒ 真实里对应的是 GQA：`kv_heads=8`、`head_dim=128` 在 VLM 文本塔和 Expert 上对齐，
所以 hidden 4096(vlm) 和 2048(expert) 不同也能共用 cache（见 real3_kv_cache.py）</sub>

**4-9** ② 里 `causal=True` 和 `causal=False` **分别用在哪**？为什么同一个 block 要两种？
<sub>考点：`PrefixVLM` 里 `causal=True`（前缀是 VLM 自回归生成的，位置 i 只能看 0..i）；
`PrefixExpert` 里 `causal=False`（64 个 waypoint 互相可见）。
后者和 4-2 是同一个理由——**轨迹是一个整体**，不是从左到右写出来的。
⇒ 真实里对应 `expert_non_causal_attention=True`</sub>

## Stage 5 — 组装

**5-1** `MiniVLA.sample` 的三行分别对应哪三个模块？
<sub>考点：cond_gen → fm.sample → action_to_traj</sub>

**5-2** `(B,64,2)` 会不会进 transformer？Transformer 吃的是什么形状？
<sub>考点：不会；吃的是 InProj 之后的 (B,64,64)</sub>

**5-3** stage5 里的 `mock_vlm` 返回随机数，为什么模型还能「跑通」？
<sub>考点：跑通≠有效；没训练时条件无意义，但数据流通了</sub>

---

## Stage 6 — 训练（flow matching）

**6-1** 训练目标 `v_target` 是什么？为什么是这个量？
<sub>考点：`x1 - x0`，即从噪声指向数据的「直线速度」</sub>

**6-2** 训练时网络「看得见」干净的 target `x1` 吗？
<sub>考点：看不见——只看到插值点 x_t；要去噪就得自己推断</sub>

**6-3** 为什么 `straight`（κ=0）模式收敛得最差？
<sub>考点：小信号 + 要求精确为 0，而 left/right 只要符号对</sub>

---

## Stage 7 — 历史轨迹作为条件

**7-1** stage7 的 condition 从哪来？和 stage6 的本质区别是什么？
<sub>考点：从历史轨迹序列「编码」；stage6 是查表（离散 embedding）</sub>

**7-2** `nn.TransformerEncoder` 自带位置编码吗？不补会怎样？
<sub>考点：不带；序列顺序失去意义（「先左后右」=「先右后左」）</sub>

---

## Stage 8 — 多模态

**8-1** 为什么「直接回归」无法表达多模态？它会输出什么？
<sub>考点：回归输出条件均值；左转 50% + 右转 50% → 直行（不存在的轨迹）</sub>

**8-2** 20 个样本的**平均 κ ≈ 0**，能说明「直行是对的」吗？
<sub>考点：不能——是两簇互相抵消，而不是「直行」</sub>

**8-3** 那几个 κ≈±0.07 的「中间样本」是怎么产生的？
<sub>考点：可能来自有限步数、模型近似或初始噪声位于两簇之间；不能据此断言存在鞍点</sub>

---

## Stage 9 — 视觉编码（真 ViT）

**9-1** ViT 输出 `(B, 17, 64)`，为什么是 17 而不是 16？
<sub>考点：1 个 CLS token + 16 个 patch token</sub>

**9-2** 为什么用随机初始化，而不加载预训练权重？
<sub>考点：输入尺寸/通道/patch 全不匹配（16×16 灰度 vs 224×224 RGB）</sub>

---

## Stage 10 — 文本编码

**10-1** 文本和图片作为输入，处理方式有什么**本质区别**？
<sub>考点：文本天生离散（查表即可）；图片连续高维（要 CNN/ViT 压）</sub>

**10-2** stage10 和 stage7 在结构上有什么关系？
<sub>考点：同一个套路——「序列 → transformer → condition」，只是输入序列不同</sub>

---

## Stage 11 — 多模态融合

**11-1** 三路（历史/图片/文本）是怎么融合的？
<sub>考点：concat 成一条序列，`(B,27,64)`</sub>

**11-2** 融合后 Expert 如何区分不同位置的 token？
<sub>考点：Expert 不接收显式模态 ID，但固定槽位和位置编码提供弱身份线索</sub>

---

## Stage 12 — 多相机

**12-1** 为什么两个相机共享同一个 ViT，而不是各用一个？
<sub>考点：省参数 + 强制学通用视觉特征</sub>

**12-2** 相机身份「用文本标签」和「用可学习 embedding」有什么区别？
<sub>考点：标签作为独立 token，先与本路视觉 token 做 attention，再与另一相机拼接</sub>

---

## Stage 13 — 自回归 CoC 生成（变长）

> stage13 现在有两个 part：
> **Part 1** cross-attention 版（生成 CoC + 条件）；**Part 2** prefix 版（同一任务，对比）

**13-1** 训练时和推理时，**condition 的长度一致吗？为什么必须一致**？变长之后怎么保证的？
<sub>考点：必须一致，否则 Expert 见到的输入分布不同；推理时把生成结果【补齐到定长】再算 condition</sub>

**13-2** 什么是 **teacher forcing**？它和「自回归生成」是什么关系？
<sub>考点：训练=并行喂真值前缀；推理=串行一个个生成；同一个任务，并行 vs 串行</sub>

**13-3** 变长生成带来**两个**要处理的问题：
（a）训练时短的推理链怎么办？（b）生成时怎么知道该停？
<sub>考点：(a) 补齐 + 用 IGNORE_INDEX 跳过 pad；(b) 见到 <eos> 停，且【逐个样本】判断
（不能等整个 batch）</sub>

**13-4** `ignore_index` 和 `key_padding_mask` **分别在解决什么**？它们能互相替代吗？
<sub>考点：ignore_index 只管【loss】（pad 不进梯度）；key_padding_mask 管【attention】
（pad 的 hidden 不被读到）。**不能互替**——pad 仍然会进 transformer、产生 hidden</sub>

**13-5** Part 2 的 prefix 版和 Part 1 的 cross-attn 版，`step_fn` 的**签名有什么不同**？说明了什么？
<sub>考点：① 是 `step_fn(x, t)` 用 `self.condition`；② 是 `step_fn(x, t, caches)`。
说明条件在 ② 里【完全通过 cache 传递】，没有单独的 condition 张量</sub>

**13-5b** Part 1 和 Part 2 是「进阶关系」还是「对等关系」？为什么留着 Part 1？
<sub>考点：**对等** —— 两种连接拓扑，不是「学会了①才轮到②」。留着 Part 1 是因为
「prefix 也能把条件传过去」这句话需要对照组；另外 cross-attn 的条件是显式张量，
stage7~12 靠它才能 `print(condition.shape)` 看清维度。
⚠️ 选 prefix 的理由是【真实代码就那么写的】，**不是②在 toy 上跑分更高**——
单一 seed、这么小的模型不构成效果比较</sub>

**13-6** `generate` 是增量的：每步只把**新 token** 喂进 CosmosReason。
那它凭什么敢不重算前缀？为什么 K/V 的前缀部分不会因此变掉？
<sub>考点：因果 mask——位置 j 的 K/V 只依赖 token 0..j，**永远看不到后面的 token**，
所以后面追加多少 token 都不影响已算好的 K/V。这正是「能 cache」的全部理由</sub>

**13-7** 变长生成会给已结束的样本补 `<pad>`，这些 `<pad>` 也进了 cache（成为 Expert 的前缀）。
Part 1 用什么屏蔽？Part 2 呢？
<sub>考点：Part 1 是 `cond_pad_mask` → cross-attn 的 `key_padding_mask`；
Part 2 是 `cache_pad_mask` → attention 里 `masked_fill(-inf)`。
**同一个问题的两个位置**（条件张量侧 / 前缀 K/V 侧），都不能靠 IGNORE_INDEX 解决</sub>

## Stage 14 — 完整输入

**14-1** stage14 比 stage13 多了什么？为什么这个「多」很重要？
<sub>考点：多了历史轨迹；真实 Alpamayo 的历史会参与条件构建和初始状态估计</sub>

---

## Stage 15 — CFG

**15-1** 写出 CFG 的公式。
<sub>考点：`v = (1-w)·v_uncond + w·v_cond`</sub>

**15-2** 为什么不能把 w>1 简单理解成「曲率放大器」？
<sub>考点：CFG 是向量场外推；轨迹曲率、稳定性和安全性都需实测，不保证单调放大</sub>

**15-3** 训练时为什么要做 condition dropout（50% 用空条件）？
<sub>考点：让网络见过「无条件」情形，推理时才能算出 v_uncond</sub>

---

# 参考答案

<details>
<summary>点开查看（先自己答完再看）</summary>

**0-1** ① `condition = VLM(...)` ② `action = fm.sample(step_fn)` ③ `action_to_traj(action)`。
顺序不能反：扩散的每一步 `step_fn` 都要读 condition，所以条件必须先算好。

**0-2** 两个 `64` 是不同的东西。`vlm_kv`/`emb` 的最后维 64 是**特征维（HIDDEN）**；
`emb`/`traj` 的中间维 64 是**时间轴（64 个 waypoint）**。巧合同值。

**0-3** 能跑（会报错，因为没定义 vlm_kv；若手动给个常量则能跑），但输出变成**随机**的——
condition 的作用是「塑造」动作分布，不是「让代码能跑」。

**1-1** ①**运动学一致**：单轮车模型按速度和航向递推，不直接产生横向瞬移；
②**低维**：64×2=128 维 vs 64×6=384 维；③更容易施加运动学约束。
但这不等于碰撞安全、舒适性或可执行性已经得到保证。

**1-2** 直线。全零动作 → `v` 恒为 v0、`θ` 恒为 0 → `x` 匀速增加、`y` 恒为 0。

**1-3** 正曲率=左转（在「车头朝 +x、y 轴指向左」的 ego 坐标系下）。取决于坐标系的 y 轴朝哪边。

**1-4** 从**历史轨迹估计**（`estimate_t0_states` 用最小二乘从历史位移反推 t0 时刻速度）。

**2-1** `x` = 当前正在被去噪的动作（初始是纯噪声）；`t` = 去噪进度 ∈ [0,1]。

**2-2** 说明这个人为提供的收缩场把噪声**逐步推向 target**；它帮助理解采样循环，
不代表网络已经学会真实数据分布。

**2-3** 控制初始噪声的尺度。调小 → 采样更稳定但**多样性降低**；关键是**训练/推理要一致**，
否则是分布漂移（我们在 stage6 踩过这个坑）。

**2-4** 方向反了 → `x` 会**远离** target，std 越变越大。

**3-1** transformer 需要足够丰富的特征才能做有意义的 attention；2 维太贫瘠。

**3-2** 普通 MLP 很难表示「随 x 快速变化」的函数；先用多频率 sin/cos 展开，MLP 只需学简单组合。

**3-3** `t` 也做 Fourier 编码，和动作特征 concat 后一起过 MLP。
必须进：因为「该走多快」取决于「去噪到哪一步了」。

**3-4** 是 **Expert**（transformer），插在 `ActionInProj` 和 `ActionOutProj` 之间。
stage3 当时还没讲，所以看起来像「InProj 直接连 OutProj」。

**4-1** self-attention：query/key/value 都是动作 token（x,x,x）。
cross-attention：query=动作 token，key/value=**条件 token**。

**4-2** 因为一条轨迹是**整体**——每个 waypoint 都该看到整条轨迹和条件，
不需要像 LLM 那样「只能看前面」（从左到右的因果约束）。

**4-3** ② 里动作 token **接在 VLM 的前缀后面**，用 self-attention 看前缀的逐层 K/V；
① 是**另开一路**，用 cross-attention 去「查」一个单独传入的条件张量。
前者是「一条数据流（续写）」，后者是「两条数据流（连接）」。

**4-4** 真实用 **②**（实测 `Qwen3VLTextModel` 第 0 层只有 `self_attn + mlp`，没有 cross-attn）。
能复用 cache 是因为 **Expert 与 VLM 文本塔结构相同**（`copy.deepcopy(vlm.config.text_config)`，
同 36 层、同 `kv_heads=8`、同 `head_dim=128`），所以 VLM 的 cache 能原样塞进去。
① 版的 Expert 是独立结构，和 VLM 对不上，只能传 hidden。

**4-5** 核心方法：**改一个输入，看输出是否变化**。
① 版：条件=随机 vs 条件=全 0 → 差异 0.1002 ≠ 0。
② 版：换一组 cache → 差异 0.0281 ≠ 0。
（反例：若条件被忽略，输出应与条件无关。）

**5-1** `MiniVLA.sample` 的三行分别对应哪三个模块？
<sub>考点：cond_gen → fm.sample → action_to_traj</sub>

**5-2** `(B,64,2)` 会不会进 transformer？Transformer 吃的是什么形状？
<sub>考点：不会；吃的是 InProj 之后的 (B,64,64)</sub>

**5-3** stage5 里的 `mock_vlm` 返回随机数，为什么模型还能「跑通」？
<sub>考点：跑通≠有效；没训练时条件无意义，但数据流通了</sub>

---

## Stage 6 — 训练（flow matching）

**6-1** 训练目标 `v_target` 是什么？为什么是这个量？
<sub>考点：`x1 - x0`，即从噪声指向数据的「直线速度」</sub>

**6-2** 训练时网络「看得见」干净的 target `x1` 吗？
<sub>考点：看不见——只看到插值点 x_t；要去噪就得自己推断</sub>

**6-3** 为什么 `straight`（κ=0）模式收敛得最差？
<sub>考点：小信号 + 要求精确为 0，而 left/right 只要符号对</sub>

---

## Stage 7 — 历史轨迹作为条件

**7-1** stage7 的 condition 从哪来？和 stage6 的本质区别是什么？
<sub>考点：从历史轨迹序列「编码」；stage6 是查表（离散 embedding）</sub>

**7-2** `nn.TransformerEncoder` 自带位置编码吗？不补会怎样？
<sub>考点：不带；序列顺序失去意义（「先左后右」=「先右后左」）</sub>

---

## Stage 8 — 多模态

**8-1** 为什么「直接回归」无法表达多模态？它会输出什么？
<sub>考点：回归输出条件均值；左转 50% + 右转 50% → 直行（不存在的轨迹）</sub>

**8-2** 20 个样本的**平均 κ ≈ 0**，能说明「直行是对的」吗？
<sub>考点：不能——是两簇互相抵消，而不是「直行」</sub>

**8-3** 那几个 κ≈±0.07 的「中间样本」是怎么产生的？
<sub>考点：可能来自有限步数、模型近似或初始噪声位于两簇之间；不能据此断言存在鞍点</sub>

---

## Stage 9 — 视觉编码（真 ViT）

**9-1** ViT 输出 `(B, 17, 64)`，为什么是 17 而不是 16？
<sub>考点：1 个 CLS token + 16 个 patch token</sub>

**9-2** 为什么用随机初始化，而不加载预训练权重？
<sub>考点：输入尺寸/通道/patch 全不匹配（16×16 灰度 vs 224×224 RGB）</sub>

---

## Stage 10 — 文本编码

**10-1** 文本和图片作为输入，处理方式有什么**本质区别**？
<sub>考点：文本天生离散（查表即可）；图片连续高维（要 CNN/ViT 压）</sub>

**10-2** stage10 和 stage7 在结构上有什么关系？
<sub>考点：同一个套路——「序列 → transformer → condition」，只是输入序列不同</sub>

---

## Stage 11 — 多模态融合

**11-1** 三路（历史/图片/文本）是怎么融合的？
<sub>考点：concat 成一条序列，`(B,27,64)`</sub>

**11-2** 融合后 Expert 如何区分不同位置的 token？
<sub>考点：Expert 不接收显式模态 ID，但固定槽位和位置编码提供弱身份线索</sub>

---

## Stage 12 — 多相机

**12-1** 为什么两个相机共享同一个 ViT，而不是各用一个？
<sub>考点：省参数 + 强制学通用视觉特征</sub>

**12-2** 相机身份「用文本标签」和「用可学习 embedding」有什么区别？
<sub>考点：标签作为独立 token，先与本路视觉 token 做 attention，再与另一相机拼接</sub>

---

## Stage 13 — 自回归 CoC 生成

> 这一段是最早的版本，完整的问题见文件开头的 13-1 ~ 13-7（含变长、pad 屏蔽、
> KV cache 增量生成、Part 2 的 prefix 对比）。下面只保留仍未过时的两条。

**13-1** 训练时和推理时，condition 的长度一致吗？
<sub>考点：必须一致，否则 Expert 见到的输入分布不同；推理时把生成结果【补齐到定长】
再算 condition（Part 1 走这条路；Part 2 用 cache，长度天然由生成过程决定）</sub>

**13-2** 什么是 teacher forcing？它和「自回归生成」是什么关系？
<sub>考点：训练=并行喂真值前缀；推理=串行一个个生成。任务是同一个。
注意 `[<bos>, r1, ..., r_{n-1}]` 这种「整条前缀一次喂进去」的写法只出现在
训练（teacher forcing）和 Part 1 补跑拿 hidden 的那次 forward 里；
`generate` 本身是增量的，每步只喂【新 token】</sub>

---

## Stage 14 — 完整输入

**14-1** stage14 比 stage13 多了什么？为什么这个「多」很重要？
<sub>考点：多了历史轨迹；真实 Alpamayo 的历史会参与条件构建和初始状态估计</sub>

---

## Stage 15 — CFG

**15-1** 写出 CFG 的公式。
<sub>考点：`v = (1-w)·v_uncond + w·v_cond`</sub>

**15-2** 为什么不能把 w>1 简单理解成「曲率放大器」？
<sub>考点：CFG 是向量场外推；轨迹曲率、稳定性和安全性都需实测，不保证单调放大</sub>

**15-3** 训练时为什么要做 condition dropout（50% 用空条件）？
<sub>考点：让网络见过「无条件」情形，推理时才能算出 v_uncond</sub>

---

# 参考答案

<details>
<summary>点开查看（先自己答完再看）</summary>

**0-1** ① `condition = VLM(...)` ② `action = fm.sample(step_fn)` ③ `action_to_traj(action)`。
顺序不能反：扩散的每一步 `step_fn` 都要读 condition，所以条件必须先算好。

**0-2** 两个 `64` 是不同的东西。`vlm_kv`/`emb` 的最后维 64 是**特征维（HIDDEN）**；
`emb`/`traj` 的中间维 64 是**时间轴（64 个 waypoint）**。巧合同值。

**0-3** 能跑（会报错，因为没定义 vlm_kv；若手动给个常量则能跑），但输出变成**随机**的——
condition 的作用是「塑造」动作分布，不是「让代码能跑」。

**1-1** ①**运动学一致**：单轮车模型按速度和航向递推，不直接产生横向瞬移；
②**低维**：64×2=128 维 vs 64×6=384 维；③更容易施加运动学约束。
但这不等于碰撞安全、舒适性或可执行性已经得到保证。

**1-2** 直线。全零动作 → `v` 恒为 v0、`θ` 恒为 0 → `x` 匀速增加、`y` 恒为 0。

**1-3** 正曲率=左转（在「车头朝 +x、y 轴指向左」的 ego 坐标系下）。取决于坐标系的 y 轴朝哪边。

**1-4** 从**历史轨迹估计**（`estimate_t0_states` 用最小二乘从历史位移反推 t0 时刻速度）。

**2-1** `x` = 当前正在被去噪的动作（初始是纯噪声）；`t` = 去噪进度 ∈ [0,1]。

**2-2** 说明这个人为提供的收缩场把噪声**逐步推向 target**；它帮助理解采样循环，
不代表网络已经学会真实数据分布。

**2-3** 控制初始噪声的尺度。调小 → 采样更稳定但**多样性降低**；关键是**训练/推理要一致**，
否则是分布漂移（我们在 stage6 踩过这个坑）。

**2-4** 方向反了 → `x` 会**远离** target，std 越变越大。

**3-1** transformer 需要足够丰富的特征才能做有意义的 attention；2 维太贫瘠。

**3-2** 普通 MLP 很难表示「随 x 快速变化」的函数；先用多频率 sin/cos 展开，MLP 只需学简单组合。

**3-3** `t` 也做 Fourier 编码，和动作特征 concat 后一起过 MLP。
必须进：因为「该走多快」取决于「去噪到哪一步了」。

**3-4** 是 **Expert**（transformer），插在 `ActionInProj` 和 `ActionOutProj` 之间。
stage3 当时还没讲，所以看起来像「InProj 直接连 OutProj」。

**4-1** self-attention：query/key/value 都是动作 token（x,x,x）。
cross-attention：query=动作 token，key/value=**条件 token**。

**4-2** 因为一条轨迹是**整体**——每个 waypoint 都该看到整条轨迹和条件，
不需要像 LLM 那样「只能看前面」（从左到右的因果约束）。

**4-3** 因为 Expert 的输入不是 token id，而是 `action_in_proj` 直接产出的 embedding，
不需要词嵌入层。

**4-4** 把条件换成不同内容，看输出是否变化（我们实测差异 0.0928 ≠ 0）；
反例：若 cross-attn 被关掉，输出应与条件无关。

**5-1** `cond_gen`（条件）→ `fm.sample`（扩散采样）→ `action_to_traj`（转轨迹）。

**5-2** 不会。Transformer 吃的是 InProj 之后的 `(B,64,64)`；`(B,64,2)` 是进 InProj 之前和 OutProj 之后的形状。

**5-3** 因为**数据流通了**——形状对、能 forward/backward。但条件无意义，所以输出也是随机的。
这正好说明 stage5 只是「框架」，stage6 才让它「有效」。

**6-1** `v_target = x1 - x0`（从噪声指向数据的直线速度）。
因为在直线插值 `x_t=(1-t)x0+t·x1` 下，沿这个速度走 1 个单位时间正好到达 x1。

**6-2** 看不见。网络只看到插值点 `x_t`、时间 `t` 和条件；
`x1` 只用来「造数据」（构造 x_t 和 v_target），不作为输入。

**6-3** ①目标 κ=0 是「小信号」，埋在 σ=1 的噪声里；②left/right 只要**符号**对，
straight 要**精确等于 0**，对残差更敏感。

**7-1** 从**历史轨迹序列**经 `ConditionEncoder`（transformer）编码而来。
stage6 是「离散指令 → 查表（`nn.Embedding`）」，stage7 是「序列 → 编码」——**从输入推导**。

**7-2** 不自带。不加位置编码，序列就变成**集合**——「先左后右」和「先右后左」对模型一样。

**8-1** 回归只能输出**条件均值**。左转 50% + 右转 50% → 平均值是 κ=0（直行），
而直行**根本不在合法选项里**。

**8-2** 不能。平均 0 是「一半左、一半右互相抵消」的结果，不是「直行是正确答案」。

**8-3** 可能是有限步数、模型近似误差或初始噪声落在两簇之间的区域；
仅凭平均曲率不能判断样本是否学到了合法模式。

**9-1** 17 = 1 个 **CLS token**（全局汇总）+ 4×4=16 个 **patch token**。

**9-2** 预训练 ViT 是 224×224、RGB、patch=16；我们的是 16×16、灰度、patch=4。
**patch embedding 那层的形状完全对不上**，而且 ImageNet 特征对这个任务也没用。

**10-1** 文本**天生是离散 token**，查表（embedding）即可；
图片是连续高维数据，必须先压（CNN/ViT）成 token。

**10-2** 同一个套路：「序列 → transformer → condition」。
stage7 输入的是历史坐标序列，stage10 输入的是文本 token 序列。

**11-1** torch.cat 成一条序列：`(B,8,64)+(B,17,64)+(B,2,64) → (B,27,64)`。

**11-2** Expert 不接收单独的「模态 ID」，对它们统一视作 token；但固定拼接顺序和位置编码
提供了弱身份线索。若要显式区分模态，应加入 modality embedding 或类型 token。

**12-1** ①省参数（参数不翻倍）；②强制 ViT 学「两种视角通用的视觉特征」。

**12-2** 本章把相机标签作为**独立 token**，先与本路视觉 token 经过 attention，再与另一相机拼接；
这样标签和图片建立了局部对应关系。它仍是 toy 结构，不等于真实模型的完整多模态序列。

**13-1** **必须一致**——否则 Expert 在训练和推理时看到的输入分布不同（我们在最早版本踩过：训练 3 个、
推理 4 个）。变长之后：推理时把【生成结果补齐到 MAX_LEN】再算 condition，
这样训练（补齐后的链）和推理（补齐后的生成）形状与含义都对得上。

**13-2** teacher forcing = 训练时**并行地**喂「正确答案的前缀」让模型猜下一个 token；
自回归生成 = 推理时**串行地**一个个猜、拼回去再猜下一个。
两者是**同一个任务**（next-token 预测），只是并行 vs 串行。

**13-3**（a）短的链**补齐**到最长，pad 位置用 `IGNORE_INDEX` 使它们不参与 loss。
（b）生成时见到 `<eos>` 就停；但要用**逐个样本的 `finished` 标志**，不能写成
`(next_tok == EOS).all()`（那是「整个 batch 都 EOS 才停」，会让提前结束的样本继续生成）。
已结束的样本后续喂 `<pad>`。

**13-4** **不能互相替代**，它们管的是两件事：
- `ignore_index`：让 pad **不参与 loss**（不算梯度）
- `key_padding_mask`：让 Expert 的 cross-attention **不读** pad 位置的 hidden
关键：pad 即使被 `ignore_index` 排除，**仍然会进 transformer、产生 hidden**，
如果不加 mask，Expert 就会读到这些（无意义的）hidden。

**13-5** ① 是 `step_fn(x, t)`（条件从 `self.condition` 取）；② 是 `step_fn(x, t, caches)`。
说明 **② 里条件完全在 cache 中传递，没有单独的 condition 张量**——这正是 prefix 方式的特征。

**14-1** stage14 比 stage13 多了什么？为什么这个「多」很重要？
<sub>考点：多了历史轨迹；真实 Alpamayo 的历史会参与条件构建和初始状态估计</sub>

---

## Stage 15 — CFG

**15-1** 写出 CFG 的公式。
<sub>考点：`v = (1-w)·v_uncond + w·v_cond`</sub>

**15-2** 为什么不能把 w>1 简单理解成「曲率放大器」？
<sub>考点：CFG 是向量场外推；轨迹曲率、稳定性和安全性都需实测，不保证单调放大</sub>

**15-3** 训练时为什么要做 condition dropout（50% 用空条件）？
<sub>考点：让网络见过「无条件」情形，推理时才能算出 v_uncond</sub>

---

# 参考答案

<details>
<summary>点开查看（先自己答完再看）</summary>

**0-1** ① `condition = VLM(...)` ② `action = fm.sample(step_fn)` ③ `action_to_traj(action)`。
顺序不能反：扩散的每一步 `step_fn` 都要读 condition，所以条件必须先算好。

**0-2** 两个 `64` 是不同的东西。`vlm_kv`/`emb` 的最后维 64 是**特征维（HIDDEN）**；
`emb`/`traj` 的中间维 64 是**时间轴（64 个 waypoint）**。巧合同值。

**0-3** 能跑（会报错，因为没定义 vlm_kv；若手动给个常量则能跑），但输出变成**随机**的——
condition 的作用是「塑造」动作分布，不是「让代码能跑」。

**1-1** ①**运动学一致**：单轮车模型按速度和航向递推，不直接产生横向瞬移；
②**低维**：64×2=128 维 vs 64×6=384 维；③更容易施加运动学约束。
但这不等于碰撞安全、舒适性或可执行性已经得到保证。

**1-2** 直线。全零动作 → `v` 恒为 v0、`θ` 恒为 0 → `x` 匀速增加、`y` 恒为 0。

**1-3** 正曲率=左转（在「车头朝 +x、y 轴指向左」的 ego 坐标系下）。取决于坐标系的 y 轴朝哪边。

**1-4** 从**历史轨迹估计**（`estimate_t0_states` 用最小二乘从历史位移反推 t0 时刻速度）。

**2-1** `x` = 当前正在被去噪的动作（初始是纯噪声）；`t` = 去噪进度 ∈ [0,1]。

**2-2** 说明这个人为提供的收缩场把噪声**逐步推向 target**；它帮助理解采样循环，
不代表网络已经学会真实数据分布。

**2-3** 控制初始噪声的尺度。调小 → 采样更稳定但**多样性降低**；关键是**训练/推理要一致**，
否则是分布漂移（我们在 stage6 踩过这个坑）。

**2-4** 方向反了 → `x` 会**远离** target，std 越变越大。

**3-1** transformer 需要足够丰富的特征才能做有意义的 attention；2 维太贫瘠。

**3-2** 普通 MLP 很难表示「随 x 快速变化」的函数；先用多频率 sin/cos 展开，MLP 只需学简单组合。

**3-3** `t` 也做 Fourier 编码，和动作特征 concat 后一起过 MLP。
必须进：因为「该走多快」取决于「去噪到哪一步了」。

**3-4** 是 **Expert**（transformer），插在 `ActionInProj` 和 `ActionOutProj` 之间。
stage3 当时还没讲，所以看起来像「InProj 直接连 OutProj」。

**4-1** self-attention：query/key/value 都是动作 token（x,x,x）。
cross-attention：query=动作 token，key/value=**条件 token**。

**4-2** 因为一条轨迹是**整体**——每个 waypoint 都该看到整条轨迹和条件，
不需要像 LLM 那样「只能看前面」（从左到右的因果约束）。

**4-3** ② 里动作 token **接在 VLM 的前缀后面**，用 self-attention 看前缀的逐层 K/V；
① 是**另开一路**，用 cross-attention 去「查」一个单独传入的条件张量。
前者是「一条数据流（续写）」，后者是「两条数据流（连接）」。

**4-4** 真实用 **②**（实测 `Qwen3VLTextModel` 第 0 层只有 `self_attn + mlp`，没有 cross-attn）。
能复用 cache 是因为 **Expert 与 VLM 文本塔结构相同**（`copy.deepcopy(vlm.config.text_config)`，
同 36 层、同 `kv_heads=8`、同 `head_dim=128`），所以 VLM 的 cache 能原样塞进去。
① 版的 Expert 是独立结构，和 VLM 对不上，只能传 hidden。

**4-5** 核心方法：**改一个输入，看输出是否变化**。
① 版：条件=随机 vs 条件=全 0 → 差异 0.1002 ≠ 0。
② 版：换一组 cache → 差异 0.0281 ≠ 0。
（反例：若条件被忽略，输出应与条件无关。）

**5-1** `MiniVLA.sample` 的三行分别对应哪三个模块？
<sub>考点：cond_gen → fm.sample → action_to_traj</sub>

**5-2** `(B,64,2)` 会不会进 transformer？Transformer 吃的是什么形状？
<sub>考点：不会；吃的是 InProj 之后的 (B,64,64)</sub>

**5-3** stage5 里的 `mock_vlm` 返回随机数，为什么模型还能「跑通」？
<sub>考点：跑通≠有效；没训练时条件无意义，但数据流通了</sub>

---

## Stage 6 — 训练（flow matching）

**6-1** 训练目标 `v_target` 是什么？为什么是这个量？
<sub>考点：`x1 - x0`，即从噪声指向数据的「直线速度」</sub>

**6-2** 训练时网络「看得见」干净的 target `x1` 吗？
<sub>考点：看不见——只看到插值点 x_t；要去噪就得自己推断</sub>

**6-3** 为什么 `straight`（κ=0）模式收敛得最差？
<sub>考点：小信号 + 要求精确为 0，而 left/right 只要符号对</sub>

---

## Stage 7 — 历史轨迹作为条件

**7-1** stage7 的 condition 从哪来？和 stage6 的本质区别是什么？
<sub>考点：从历史轨迹序列「编码」；stage6 是查表（离散 embedding）</sub>

**7-2** `nn.TransformerEncoder` 自带位置编码吗？不补会怎样？
<sub>考点：不带；序列顺序失去意义（「先左后右」=「先右后左」）</sub>

---

## Stage 8 — 多模态

**8-1** 为什么「直接回归」无法表达多模态？它会输出什么？
<sub>考点：回归输出条件均值；左转 50% + 右转 50% → 直行（不存在的轨迹）</sub>

**8-2** 20 个样本的**平均 κ ≈ 0**，能说明「直行是对的」吗？
<sub>考点：不能——是两簇互相抵消，而不是「直行」</sub>

**8-3** 那几个 κ≈±0.07 的「中间样本」是怎么产生的？
<sub>考点：可能来自有限步数、模型近似或初始噪声位于两簇之间；不能据此断言存在鞍点</sub>

---

## Stage 9 — 视觉编码（真 ViT）

**9-1** ViT 输出 `(B, 17, 64)`，为什么是 17 而不是 16？
<sub>考点：1 个 CLS token + 16 个 patch token</sub>

**9-2** 为什么用随机初始化，而不加载预训练权重？
<sub>考点：输入尺寸/通道/patch 全不匹配（16×16 灰度 vs 224×224 RGB）</sub>

---

## Stage 10 — 文本编码

**10-1** 文本和图片作为输入，处理方式有什么**本质区别**？
<sub>考点：文本天生离散（查表即可）；图片连续高维（要 CNN/ViT 压）</sub>

**10-2** stage10 和 stage7 在结构上有什么关系？
<sub>考点：同一个套路——「序列 → transformer → condition」，只是输入序列不同</sub>

---

## Stage 11 — 多模态融合

**11-1** 三路（历史/图片/文本）是怎么融合的？
<sub>考点：concat 成一条序列，`(B,27,64)`</sub>

**11-2** 融合后 Expert 如何区分不同位置的 token？
<sub>考点：Expert 不接收显式模态 ID，但固定槽位和位置编码提供弱身份线索</sub>

---

## Stage 12 — 多相机

**12-1** 为什么两个相机共享同一个 ViT，而不是各用一个？
<sub>考点：省参数 + 强制学通用视觉特征</sub>

**12-2** 相机身份「用文本标签」和「用可学习 embedding」有什么区别？
<sub>考点：标签作为独立 token，先与本路视觉 token 做 attention，再与另一相机拼接</sub>

---

## Stage 13 — 自回归 CoC 生成

> 这一段是最早的版本，完整的问题见文件开头的 13-1 ~ 13-7（含变长、pad 屏蔽、
> KV cache 增量生成、Part 2 的 prefix 对比）。下面只保留仍未过时的两条。

**13-1** 训练时和推理时，condition 的长度一致吗？
<sub>考点：必须一致，否则 Expert 见到的输入分布不同；推理时把生成结果【补齐到定长】
再算 condition（Part 1 走这条路；Part 2 用 cache，长度天然由生成过程决定）</sub>

**13-2** 什么是 teacher forcing？它和「自回归生成」是什么关系？
<sub>考点：训练=并行喂真值前缀；推理=串行一个个生成。任务是同一个。
注意 `[<bos>, r1, ..., r_{n-1}]` 这种「整条前缀一次喂进去」的写法只出现在
训练（teacher forcing）和 Part 1 补跑拿 hidden 的那次 forward 里；
`generate` 本身是增量的，每步只喂【新 token】</sub>

---

## Stage 14 — 完整输入

**14-1** stage14 比 stage13 多了什么？为什么这个「多」很重要？
<sub>考点：多了历史轨迹；真实 Alpamayo 的历史会参与条件构建和初始状态估计</sub>

---

## Stage 15 — CFG

**15-1** 写出 CFG 的公式。
<sub>考点：`v = (1-w)·v_uncond + w·v_cond`</sub>

**15-2** 为什么不能把 w>1 简单理解成「曲率放大器」？
<sub>考点：CFG 是向量场外推；轨迹曲率、稳定性和安全性都需实测，不保证单调放大</sub>

**15-3** 训练时为什么要做 condition dropout（50% 用空条件）？
<sub>考点：让网络见过「无条件」情形，推理时才能算出 v_uncond</sub>

---

# 参考答案

<details>
<summary>点开查看（先自己答完再看）</summary>

**0-1** ① `condition = VLM(...)` ② `action = fm.sample(step_fn)` ③ `action_to_traj(action)`。
顺序不能反：扩散的每一步 `step_fn` 都要读 condition，所以条件必须先算好。

**0-2** 两个 `64` 是不同的东西。`vlm_kv`/`emb` 的最后维 64 是**特征维（HIDDEN）**；
`emb`/`traj` 的中间维 64 是**时间轴（64 个 waypoint）**。巧合同值。

**0-3** 能跑（会报错，因为没定义 vlm_kv；若手动给个常量则能跑），但输出变成**随机**的——
condition 的作用是「塑造」动作分布，不是「让代码能跑」。

**1-1** ①**运动学一致**：单轮车模型按速度和航向递推，不直接产生横向瞬移；
②**低维**：64×2=128 维 vs 64×6=384 维；③更容易施加运动学约束。
但这不等于碰撞安全、舒适性或可执行性已经得到保证。

**1-2** 直线。全零动作 → `v` 恒为 v0、`θ` 恒为 0 → `x` 匀速增加、`y` 恒为 0。

**1-3** 正曲率=左转（在「车头朝 +x、y 轴指向左」的 ego 坐标系下）。取决于坐标系的 y 轴朝哪边。

**1-4** 从**历史轨迹估计**（`estimate_t0_states` 用最小二乘从历史位移反推 t0 时刻速度）。

**2-1** `x` = 当前正在被去噪的动作（初始是纯噪声）；`t` = 去噪进度 ∈ [0,1]。

**2-2** 说明这个人为提供的收缩场把噪声**逐步推向 target**；它帮助理解采样循环，
不代表网络已经学会真实数据分布。

**2-3** 控制初始噪声的尺度。调小 → 采样更稳定但**多样性降低**；关键是**训练/推理要一致**，
否则是分布漂移（我们在 stage6 踩过这个坑）。

**2-4** 方向反了 → `x` 会**远离** target，std 越变越大。

**3-1** transformer 需要足够丰富的特征才能做有意义的 attention；2 维太贫瘠。

**3-2** 普通 MLP 很难表示「随 x 快速变化」的函数；先用多频率 sin/cos 展开，MLP 只需学简单组合。

**3-3** `t` 也做 Fourier 编码，和动作特征 concat 后一起过 MLP。
必须进：因为「该走多快」取决于「去噪到哪一步了」。

**3-4** 是 **Expert**（transformer），插在 `ActionInProj` 和 `ActionOutProj` 之间。
stage3 当时还没讲，所以看起来像「InProj 直接连 OutProj」。

**4-1** self-attention：query/key/value 都是动作 token（x,x,x）。
cross-attention：query=动作 token，key/value=**条件 token**。

**4-2** 因为一条轨迹是**整体**——每个 waypoint 都该看到整条轨迹和条件，
不需要像 LLM 那样「只能看前面」（从左到右的因果约束）。

**4-3** 因为 Expert 的输入不是 token id，而是 `action_in_proj` 直接产出的 embedding，
不需要词嵌入层。

**4-4** 把条件换成不同内容，看输出是否变化（我们实测差异 0.0928 ≠ 0）；
反例：若 cross-attn 被关掉，输出应与条件无关。

**5-1** `cond_gen`（条件）→ `fm.sample`（扩散采样）→ `action_to_traj`（转轨迹）。

**5-2** 不会。Transformer 吃的是 InProj 之后的 `(B,64,64)`；`(B,64,2)` 是进 InProj 之前和 OutProj 之后的形状。

**5-3** 因为**数据流通了**——形状对、能 forward/backward。但条件无意义，所以输出也是随机的。
这正好说明 stage5 只是「框架」，stage6 才让它「有效」。

**6-1** `v_target = x1 - x0`（从噪声指向数据的直线速度）。
因为在直线插值 `x_t=(1-t)x0+t·x1` 下，沿这个速度走 1 个单位时间正好到达 x1。

**6-2** 看不见。网络只看到插值点 `x_t`、时间 `t` 和条件；
`x1` 只用来「造数据」（构造 x_t 和 v_target），不作为输入。

**6-3** ①目标 κ=0 是「小信号」，埋在 σ=1 的噪声里；②left/right 只要**符号**对，
straight 要**精确等于 0**，对残差更敏感。

**7-1** 从**历史轨迹序列**经 `ConditionEncoder`（transformer）编码而来。
stage6 是「离散指令 → 查表（`nn.Embedding`）」，stage7 是「序列 → 编码」——**从输入推导**。

**7-2** 不自带。不加位置编码，序列就变成**集合**——「先左后右」和「先右后左」对模型一样。

**8-1** 回归只能输出**条件均值**。左转 50% + 右转 50% → 平均值是 κ=0（直行），
而直行**根本不在合法选项里**。

**8-2** 不能。平均 0 是「一半左、一半右互相抵消」的结果，不是「直行是正确答案」。

**8-3** 可能是有限步数、模型近似误差或初始噪声落在两簇之间的区域；
仅凭平均曲率不能判断样本是否学到了合法模式。

**9-1** 17 = 1 个 **CLS token**（全局汇总）+ 4×4=16 个 **patch token**。

**9-2** 预训练 ViT 是 224×224、RGB、patch=16；我们的是 16×16、灰度、patch=4。
**patch embedding 那层的形状完全对不上**，而且 ImageNet 特征对这个任务也没用。

**10-1** 文本**天生是离散 token**，查表（embedding）即可；
图片是连续高维数据，必须先压（CNN/ViT）成 token。

**10-2** 同一个套路：「序列 → transformer → condition」。
stage7 输入的是历史坐标序列，stage10 输入的是文本 token 序列。

**11-1** torch.cat 成一条序列：`(B,8,64)+(B,17,64)+(B,2,64) → (B,27,64)`。

**11-2** Expert 不接收单独的「模态 ID」，对它们统一视作 token；但固定拼接顺序和位置编码
提供了弱身份线索。若要显式区分模态，应加入 modality embedding 或类型 token。

**12-1** ①省参数（参数不翻倍）；②强制 ViT 学「两种视角通用的视觉特征」。

**12-2** 本章把相机标签作为**独立 token**，先与本路视觉 token 经过 attention，再与另一相机拼接；
这样标签和图片建立了局部对应关系。它仍是 toy 结构，不等于真实模型的完整多模态序列。

**13-1** **曾经不一致**：训练 `[bos,turn,left]`=3 个，推理生成 4 个 → `(B,3,64)` vs `(B,4,64)`。
已修：两边都取「推理前缀去 bos」= 2 个 → `(B,2,64)`。

**13-2** teacher forcing = 训练时**并行地**喂「正确答案的前缀」让模型猜下一个 token；
自回归生成 = 推理时**串行地**一个个猜、拼回去再猜下一个。
**两者是同一个任务**（next-token 预测），只是并行 vs 串行。

**13-3** 让推理的输入与训练的输入**对齐**：训练输入是 `[bos,turn,left]`（不含 eos），
所以推理也要去掉末尾的 eos，得到的 condition 长度才一致。

**14-1** 多了**历史轨迹**（和图片一起拼进 CosmosReason 输入）。
重要：历史轨迹会同时影响 VLM 条件和积分初速度；real6 的单 clip 置零实验不能单独量化其模态重要性。

**15-1** `v_guided = (1-w) · v_unconditional + w · v_conditional`。

**15-2** CFG 是对条件/无条件向量场的**外推**，可能产生训练数据之外的行为；
它不保证曲率单调放大，也不保证更安全或更准确。

**15-3** 为了让网络**见过「无指令」是什么样**，推理时才能算出 `v_unconditional`，
否则 CFG 公式里的无条件项无从得到。

</details>
