# Alpamayo 1.5 VLA 学习指南（由浅入深）

面向：熟悉 3D 检测 / BEV、有一定 LLM 基础、但对 VLA 不太熟的读者。
本文用这份代码库作为具体例子，把 VLA 的核心思想讲透。所有文件路径都相对 `src/alpamayo1_5/`。

---

## 0. 一句话心智模型

> Alpamayo 1.5 是一个「**视觉-语言-动作**」模型：**VLM 看多相机画面 + 历史轨迹，先用自然语言写一段因果链推理（CoC），再把这个推理过程中产生的隐状态（KV cache）作为条件，喂给一个 diffusion 模型，去「去噪」出一条未来轨迹（64 个 waypoints，6.4 秒）。**

一句话拆开，对应三个模块：

| 模块 | 代码位置 | 角色 |
|---|---|---|
| **VLM 主干**（Vision+Language） | `models/base_model.py` 里的 `self.vlm`，实际是 Qwen3-VL-8B | 看图、读历史，写推理 |
| **动作专家**（Action expert） | `models/alpamayo1_5.py` 里的 `self.expert` | 一个小型 text-only transformer，负责「去噪」动作 |
| **动作空间 + 扩散** | `action_space/` + `diffusion/flow_matching.py` | 把「轨迹」参数化成连续动作，再用 flow matching 采样 |

---

## 1. 端到端数据流（先建立全局感）

入口是 `test_inference.py` → `sample_trajectories_from_data_with_vlm_rollout`（`models/alpamayo1_5.py:218`）。按顺序：

```
输入
├─ 多相机图片: 4 相机 × 4 时间帧 = 16 张图
├─ 历史轨迹: 1.6s × 16 个位姿 (xyz + 旋转)
└─ (可选) 导航指令文本

  ↓ 1) helper.create_message 拼 prompt (helper.py:77)
  ↓    图片 → image token；历史轨迹 → 占位符 <|traj_history|>×48
  ↓    文本: "output the chain-of-thought reasoning... then output the future trajectory"

  ↓ 2) fuse_traj_tokens (base_model.py:172)
  ↓    把历史轨迹编码成 16 个离散 token，替换掉占位符

  ↓ 3) VLM 自回归生成 (alpamayo1_5.py:289)
  ↓    生成 CoC 推理文本，直到吐出 <|traj_future_start|> 就停
  ↓    此时 VLM 的 KV cache = "推理状态"，是关键产物

  ↓ 4) 找到 offset（<traj_future_start> 之后的位置）(alpamayo1_5.py:309)

  ↓ 5) 扩散采样循环 (alpamayo1_5.py:332 step_fn + :368 diffusion.sample)
  ↓    从高斯噪声 x 出发，每步用 expert 算向量场 v，欧拉积分 x += dt·v
  ↓    expert 通过 cross-attention 读第 3 步的 VLM KV cache

  ↓ 6) action_to_traj (alpamayo1_5.py:384)
  ↓    把采样的动作 (accel, curvature) 积分成 (xyz, rotation) 轨迹

输出: pred_xyz (B, num_samples, 1, 64, 3), pred_rot, CoC 文本
```

**先记住这条主链**，下面每一层都是往这条链上填细节。

---

## 2. 动作空间：轨迹怎么表示（离你最近的模块）

你熟悉 BEV 和 motion prediction，这里最相关的是 `action_space/unicycle_accel_curvature.py`。

**核心思想：不直接输出 xyz 坐标，而是输出「单轮车运动学模型」的控制量——加速度 accel 和曲率 curvature。**

- 动作维度 = `(n_waypoints=64, 2)`，即每个 0.1s 时间步一个 `(accel, curvature)`（`get_action_space_dims` → `unicycle_accel_curvature.py:100`）。
- `action_to_traj`（`:307`）把这个动作**积分**成轨迹：
  - 加速度 → 速度（cumsum）
  - 速度 × 曲率 → 航向角 θ（cumsum）
  - 速度 × cos/sin(θ) → x, y（cumsum）
- 反方向 `traj_to_action`（`:234`）用带 Tikhonov 正则的最小二乘从轨迹反解出 (accel, curvature)。

**为什么这样设计（这是 planning 的经典思路）**：
1. **物理合理性**：单轮车模型保证输出轨迹可被车辆执行（平滑、符合运动学），不像直接回归 64 个 xyz 可能产生抖动。
2. **低维**：64×2 = 128 维，比 64×3(xyz)+64×3(rot) 少很多，扩散模型更好学。
3. **归一化**：accel 和 curvature 都标准化到均值 0、方差 1（`accel_std`/`curvature_std`），便于网络训练。

这跟你做 BEV 时「输出是 BEV 栅格里的目标框」是同一个哲学：**选一个好的输出表示，比硬拟合原始数据更重要。**

---

## 3. VLM 主干：因果链推理（你熟悉的 LLM 部分）

VLM 是 Qwen3-VL-8B-Instruct（`base_model.py:376` `_initialize_qwenvl3_vlm`）。它就是个标准的多模态 LLM，但做了两件「VLA 特化」的事：

**3.1 加了轨迹专用 token**（`base_model.py:347`）
- 加了 `traj_vocab_size=768` 个离散 token：`<i0>` ... `<i767>`。
- 加了特殊 token：`<|traj_history_start|>`、`<|traj_future_start|>`、`<|cot_start|>` 等（`TRAJ_TOKEN` 字典，`base_model.py:40`）。

**3.2 历史轨迹被「离散化」进 prompt**（`base_model.py:95` `tokenize_history_trajectory` + `delta_tokenizer.py:47`）
- `DeltaTrajectoryTokenizer.encode`：把连续的历史轨迹变成**相邻 waypoint 之间的位移 Δ**，然后按 1000 个 bin 量化成整数 token。
- 这就是**「连续量 → 离散 token」的桥梁**：LLM 只能处理离散 token，所以连续轨迹必须量化。

**3.3 推理（CoC）就是普通文本生成**
- prompt 结尾是 `<|cot_start|>`（`helper.py:140`），引导 VLM 输出推理。
- 生成直到 `<|traj_future_start|>`（这是 CoC 的 EOS，`alpamayo1_5.py:279`）。
- 注意 `ExpertLogitsProcessor`（`alpamayo1_5.py:48`）：生成 CoC 时，把轨迹 token 的 logits 全部设为 -inf，**强制 VLM 专注写推理、不掺和轨迹 token**。

到这里为止，都还是你熟悉的 LLM 玩法。真正的「VLA 灵魂」在下一节。

---

## 4. 核心机制：语言如何「条件」动作（VLA 的灵魂，重点读）

**问题**：LLM 只会吐 token，怎么让它「指导」连续轨迹的生成？

**答案（本代码的做法）**：**不靠 LLM 吐轨迹 token，而是把 LLM 生成 CoC 后的隐状态（KV cache）当作条件，喂给一个 diffusion 专家去做「去噪」。**

具体拆解（`sample_trajectories_from_data_with_vlm_rollout`）：

1. VLM 生成 CoC 后，`prompt_cache = vlm_outputs.past_key_values`（`alpamayo1_5.py:304`）——这就是「看图 + 历史 + 推理」之后的完整隐状态。

2. 定义 `step_fn(x, t)`（`alpamayo1_5.py:332`），这是扩散采样每一步要做的事：
   ```
   噪声动作 x (B, 64, 2) + 时间步 t
     → action_in_proj: Fourier 编码 + MLP → (B, 64, hidden)  "伪 token embedding"
     → self.expert: 拿着这些 embedding，cross-attention 到 VLM 的 KV cache
     → action_out_proj: 隐状态 → 预测向量场 v (B, 64, 2)
   ```

3. `self.expert` 是**非因果**的（`expert_non_causal_attention=True`，`alpamayo1_5.py:328` `is_causal=False`），意思是所有 waypoint token 互相都能看到——这跟自回归 LLM 的因果注意力正好相反，因为轨迹去噪不需要「从左到右」的因果性。

**这就是 VLA 和「LLM + 单独规划模块」的本质区别**：
- 不是「LLM 输出一个语义命令，另一个模块执行」；
- 而是「LLM 的**连续隐状态**直接作为条件信号，端到端地塑造轨迹分布」。

用你的语言说：这有点像 BEV 里「感知特征 → 检测头」的接力，只不过这里的「特征」是 LLM 的 KV cache，「检测头」是 diffusion 专家。

---

## 5. 扩散动作头：Flow Matching（新知识，但直觉可迁移）

`diffusion/flow_matching.py`。为什么用扩散/流匹配，而不是直接回归？

- **多模态性**：同一个场景可能有多条合理轨迹（路口左转还是右转）。直接回归会「平均」掉这些模式；扩散能表达**分布**。
- **Flow Matching** 学的是一个**向量场** v(x, t)，把噪声「推」向数据分布。推理时从 `x ~ N(0, temp²)` 出发，欧拉积分 `x += dt·v`（`flow_matching.py:138 _euler`），默认 10 步。
- 有**无分类器引导（CFG）**的版本：`_guided_v`（`:114`）把「有导航指令」和「无导航指令」两个向量场按权重混合，强化导航指令的效果（对应 nav notebook 里的 CFG 部分）。

**一个值得注意的细节**：时间步 t 通过 `action_in_proj` 里的 Fourier 编码注入（`action_in_proj.py:148`），这样网络知道「当前噪声有多大」，这是扩散模型的标准做法。

---

## 6. 训练 vs 推理：Delta Tokenizer 的两种用法（理解完整图景）

你可能注意到：推理时**未来轨迹走 diffusion（连续），但历史轨迹走 delta tokenizer（离散）**。为什么？

- **历史轨迹 → 离散 token**（`delta_tokenizer.encode`）：因为它要进 VLM 的 **prompt**（LLM 只吃 token）。这是「条件」。
- **未来轨迹 → 连续 action + diffusion**（推理时）：因为未来轨迹是**要生成的多模态分布**，用离散 token 自回归生成会慢且有累积误差，用 diffusion 一步到位更好。

（训练时，`traj_tokenizer`（`delta_tokenizer` 的 768-bin 版本）也用于给未来轨迹打 token 做监督；但**推理时 release 模型走的是 diffusion 路径**，这点从 `sample_trajectories...` 全程没碰 `traj_tokenizer` 可以看出。）

---

## 7. 关键设计取舍（面试/复现常问）

1. **为什么专家（expert）是 text-only 且删掉 embed_tokens？**（`alpamayo1_5.py:105`）——因为它的输入不是 token id，而是 `action_in_proj` 直接产出的 embedding，所以不需要 embedding 层。它是「动作去噪器」，不是 LLM。

2. **为什么 expert 用非因果注意力？**——轨迹去噪需要全局信息（每个 waypoint 都该看到整条轨迹 + 条件），不需要自回归的因果约束。

3. **为什么动作用 (accel, curvature) 而不是 (v, steering) 或原始 xyz？**——见第 2 节：物理合理 + 低维 + 平滑。

4. **CoC 推理的价值**：不只是「解释性」，它让 VLM 先把场景「想清楚」，产出的 KV cache 隐式编码了推理结果，从而更准确地条件化轨迹。（对应 README 说的 "bridging reasoning and action"。）

---

## 8. 推荐阅读顺序 + 动手实验

**阅读顺序（每个文件 10~30 分钟）：**

1. `action_space/action_space.py`（94 行，抽象接口）→ `unicycle_accel_curvature.py` 的 `action_to_traj`/`traj_to_action`（理解动作表示）
2. `models/alpamayo1_5.py` 的 `sample_trajectories_from_data_with_vlm_rollout`（218~405 行，主流程，反复读）
3. `models/base_model.py` 的 `fuse_traj_tokens` + `tokenize_history_trajectory`（历史轨迹怎么进 prompt）
4. `diffusion/flow_matching.py`（扩散采样，很短）
5. `models/delta_tokenizer.py`（连续↔离散的桥梁）
6. `models/action_in_proj.py`（动作→token embedding 的投影）
7. `helper.py` 的 `create_message`（prompt 模板长啥样）

**动手实验（建议按序）：**

1. 跑 `test_inference.py`，单步调试 `sample_trajectories_from_data_with_vlm_rollout`，在 `step_fn` 里打印 `x`、`v` 的 shape，感受扩散的迭代过程。
2. 把 `num_traj_samples` 改成 4（在有显存的卡上），看同一个场景多采样的轨迹差异——直观感受「分布 vs 单点回归」。
3. 单独调 `UnicycleAccelCurvatureActionSpace.action_to_traj`，输入一个手工构造的 (accel, curvature) 动作，看积分出来的轨迹——理解动作空间。
4. 读 `nav_utils.compare_nav_conditions`，看「有指令 / 无指令 / 反向指令」三组轨迹怎么对比——理解导航条件化。
5. 想深入扩散：对照 flow matching 论文（Flow Matching for Generative Modeling, arXiv:2210.02747；Guided Flows, arXiv:2311.13443，代码注释里也引了）。

---

## 附：核心文件速查表

| 文件 | 行数 | 一句话 |
|---|---|---|
| `models/alpamayo1_5.py` | 699 | **主流程**：VLM 生成 → expert 去噪 → 转轨迹 |
| `models/base_model.py` | 505 | VLM 加载 + 轨迹 token 融合 + `generate_text` |
| `models/delta_tokenizer.py` | 216 | 连续轨迹 ↔ 离散 token（1000 bin） |
| `models/action_in_proj.py` | 166 | 噪声动作 + 时间步 → expert 输入 embedding |
| `models/token_utils.py` | 253 | token 处理工具（EOS 停止、文本抽取） |
| `diffusion/flow_matching.py` | 196 | Flow Matching 采样（欧拉积分） |
| `action_space/unicycle_accel_curvature.py` | 389 | (accel, curvature) ↔ 轨迹 |
| `action_space/utils.py` | 513 | 求解器（带正则的最小二乘） |
| `geometry/rotation.py` | 246 | 旋转矩阵/航向角工具 |
| `helper.py` | 219 | prompt 构造 |
| `nav_utils.py` | 269 | 导航条件化 + 对比 |
