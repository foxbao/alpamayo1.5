"""Stage 13: 自回归 CoC 生成 —— 迷你 Cosmos-Reason Backbone

【目的】理解 VLA 里「先推理、再行动」的那一步（CoC = Chain-of-Causation）：
  - 用因果 transformer 自回归【生成】一段推理文本，再拿它的隐状态当动作条件
  - 三个新机制：
      ① **因果 mask** —— 位置 i 只能看 0..i，区别于 Expert 的非因果 self-attn
      ② **teacher forcing** 训练 —— 输入 [<bos>, r1, ..., r_{n-1}]，预测 [r1, ..., <eos>]
      ③ **变长生成** —— 三条推理链长度不同（6/6/3），生成时遇到 <eos> 就停
  - 变长带来的两个配套机制：
      · padding —— 训练时把短链补齐，pad 位置用 IGNORE_INDEX 跳过，不算 loss
      · 按 EOS 停止 —— 不是固定步数；已结束的样本继续喂 <pad>，避免污染后续生成
  - 一个必须记住的坑：**IGNORE_INDEX 只让 pad 不参与 loss，并不能阻止 pad 进入
    transformer**。pad 仍会产生 hidden、仍会被 cross-attention 读到 ——
    要真正屏蔽必须在 attention 层用 mask（本 stage 的 cond_pad_mask / cache_pad_mask）
  - 训练损失 = cot_loss（语言建模）+ fm_loss（流匹配），两个任务联合训练

⚠️ 关键区分：本 stage 的文本是【模型生成的推理】，**不是**输入指令。

    stage10 的文本:  输入（外部给定的指令，如 "turn left in 10m"）
    stage13 的文本:  输出（模型看完图后【自己写出来】的推理）
                     如 "shift left due to curve" —— 模仿真实 CoC 的「动作 + due to + 原因」

    真实 Alpamayo 里两者都存在且角色不同：
        导航指令（输入）→ 影响推理怎么写；CoC 推理（输出）→ 其隐状态影响轨迹怎么出

★ Part 1 / Part 2 是【同一个任务、两种连接方式】的对照实验：
   Part 1：cross-attn 版 Expert，条件作为独立的 hidden 张量传入。
           因为 cross-attn 要的是 hidden 而不是 K/V，**必须再跑一次完整 forward**。
   Part 2：prefix 版 —— **Alpamayo 的真实做法**，
           直接复用 `generate()` 留下的 cache（里面含 CoC），**不需要额外 forward**，
           而且 `step_fn` 的签名里根本没有 condition 参数。
   这正是真实代码的结构：
       prompt_cache = vlm_outputs.past_key_values     # 生成时留下的缓存
       expert(..., past_key_values=prompt_cache)      # Expert 直接复用
   机制细节与计算量对比见 `exp_kv_cache.py`；KV cache 为什么必需见 `real3_kv_cache.py`。

【简化】
  - 输入只有【图片】，省略了历史轨迹（完整版见 stage14_complete.py）、
    多相机、多帧、导航指令
  - 词表只有 11 个 token（3 special + 8 实词），推理链由人工写死 3 条；
    真实词表约 15 万，CoC 是自然语言且长度/内容自由
  - CosmosReason 3 层、hidden 64，**随机初始化**；真实是预训练 Cosmos-Reason2-8B（30+ 层）
  - 贪心解码（argmax）；真实用 temperature=0.6 + top_p 采样
  - 没有 <|cot_start|>/<|cot_end|> 等 special token 框架，没有 logits processor
    （真实会屏蔽掉 4000 个离散轨迹 token，避免 CoC 生成到它们）
  - 语义监督是假的：CoC 文本和轨迹方向只是人为配对，**不证明模型真的「按推理行动」**
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
    build_vit, CosmosReason, CacheStack,
)

KAPPA = 0.1
IMG_SIZE = 16
VISUAL_TOKENS = 17   # ViT 输出：1 CLS + 16 patch
IGNORE_INDEX = -100  # 和真实代码一致（sft_base_model.py:35）

VOCAB = ["<bos>", "<eos>", "<pad>",
         "shift", "left", "right", "hold", "course", "due", "to", "curve"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BOS_ID, EOS_ID, PAD_ID = V["<bos>"], V["<eos>"], V["<pad>"]

# 三条 CoC 推理链，长度【不同】（真实 CoC 本来就是变长的）
CHAINS = [
    [V["shift"], V["left"], V["due"], V["to"], V["curve"], EOS_ID],    # 6 个
    [V["shift"], V["right"], V["due"], V["to"], V["curve"], EOS_ID],   # 6 个
    [V["hold"], V["course"], EOS_ID],                                  # 3 个 ← 更短
]
MAX_LEN = max(len(c) for c in CHAINS)

# 补齐成张量：输入用 <pad>，标签用 IGNORE_INDEX
REASONING_IN = torch.full((3, MAX_LEN), PAD_ID, dtype=torch.long)
REASONING_LAB = torch.full((3, MAX_LEN), IGNORE_INDEX, dtype=torch.long)
for _i, _c in enumerate(CHAINS):
    REASONING_IN[_i, : len(_c)] = torch.tensor(_c)
    REASONING_LAB[_i, : len(_c)] = torch.tensor(_c)


def make_image(mode):
    img = torch.zeros(1, IMG_SIZE, IMG_SIZE)
    for i in range(IMG_SIZE):
        if mode == 0:
            col = max(0, 8 - i // 2)
        elif mode == 1:
            col = min(IMG_SIZE - 1, 8 + i // 2)
        else:
            col = 8
        for d in range(3):
            c = min(IMG_SIZE - 1, max(0, col + d - 1))
            img[0, i, c] = 1.0
    return img


IMAGES = torch.stack([make_image(0), make_image(1), make_image(2)])  # (3,1,16,16)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = build_vit()
        self.cosmos = CosmosReason(VOCAB_SIZE)
        self.in_proj = ActionInProj()
        self.expert = CrossAttnExpert()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()
        self.condition = None
        self.cond_pad_mask = None

    def step_fn(self, x, t):
        emb = self.in_proj(x, t)
        h = self.expert(emb, self.condition, self.cond_pad_mask)   # ← 屏蔽 <pad>
        return self.out_proj(h)

    def _condition_from_hidden(self, h, text_in):
        """从【已经算好的】h 里取出推理部分的隐状态（去掉 <bos>），并标记 <pad>。

        同时设置 `cond_pad_mask`：标记哪些位置是 <pad>，供 Expert 在 cross-attention
        里屏蔽掉（`ignore_index` 只管 loss，管不到 attention）。
        """
        self.cond_pad_mask = text_in[:, 1:] == PAD_ID         # (B, T) True=pad
        return h[:, VISUAL_TOKENS + 1 :]                      # 去掉 <bos>

    def _reasoning_hidden(self, visual, text_in):
        """给定 [<bos>, r1, r2, ...]（已补齐到定长），返回推理部分的隐状态（去掉 <bos>）。

        训练和推理都走这一条路径，保证 condition 的长度和含义一致。
        """
        h, _ = self.cosmos(visual, text_in)
        return self._condition_from_hidden(h, text_in)

    def cache_pad_mask(self, text_ids):
        """前缀里哪些位置是 <pad>（变长生成给已结束样本补出来的）→ 交给 Expert 屏蔽。

        和 Part 1 的 `cond_pad_mask` 是同一件事在【前缀侧】的翻版：
        补出来的 <pad> 同样进了 cache，不屏蔽就会被 Expert 当成条件读进去。
        """
        B = text_ids.shape[0]
        return torch.cat([torch.zeros(B, VISUAL_TOKENS, dtype=torch.bool),
                          text_ids == PAD_ID], dim=1)

    def generate(self, image, max_len=MAX_LEN):
        """自回归生成推理（**增量 + KV cache**）：遇到 <eos> 就停（变长）。

        返回 `(text_ids, caches, cache_pad_mask)`。
        caches = [视觉 + <bos> + 生成内容] 的逐层 K/V，就是真实 VLM 生成时留下的
        `past_key_values` —— Part 2 直接拿它当条件，不用再跑第二次 forward。
        """
        B = image.shape[0]
        visual = self.vit(image).last_hidden_state            # (B,17,64)
        text_ids = torch.full((B, 1), BOS_ID, dtype=torch.long)  # 从 <bos> 开始
        finished = torch.zeros(B, dtype=torch.bool)          # 逐个样本记录"是否已吐 eos"
        # prefill：整条前缀（视觉 + <bos>）一次算完，留下逐层 K/V
        h, caches = self.cosmos(visual, text_ids)
        for _ in range(max_len):
            next_tok = self.cosmos.head(h[:, -1]).argmax(-1, keepdim=True)  # 贪心取最大
            # 已结束的样本继续喂 <pad>（不能再让它生成，否则会污染）
            next_tok = torch.where(
                finished[:, None], torch.full_like(next_tok, PAD_ID), next_tok
            )
            text_ids = torch.cat([text_ids, next_tok], dim=1)
            finished |= (next_tok.squeeze(-1) == EOS_ID)
            # ★ 增量：只把【新 token】喂进去，复用 cache —— 不重算整条前缀
            h, caches = self.cosmos(visual, next_tok, caches)
            if bool(finished.all()):
                break                                        # ← 变长：全 batch 都停了才退出
        self._visual = visual                                # 供 Part 1 取隐状态用
        return text_ids, caches, self.cache_pad_mask(text_ids)

    def sample(self, image):
        """Part 1（cross-attn）：生成 CoC → 取隐状态当条件 → 轨迹。"""
        text_ids, _, _ = self.generate(image)
        # Part 1 的 Expert 要的是【隐状态】（不是 K/V），所以还得跑一次完整 forward。
        # Part 2 就不用了——直接用 generate 产出的 cache。
        B = image.shape[0]
        gen = text_ids[:, 1:]
        if gen.shape[1] < MAX_LEN:
            gen = torch.cat([gen, torch.full((B, MAX_LEN - gen.shape[1]), PAD_ID,
                                             dtype=torch.long)], dim=1)
        text_in = torch.cat([torch.full((B, 1), BOS_ID, dtype=torch.long), gen], dim=1)
        self.condition = self._reasoning_hidden(self._visual, text_in)
        action = self.fm.sample(self.step_fn, batch_size=B)
        traj = self.action_space.action_to_traj(action)
        return traj, text_ids


def train(model, opt, target_actions, n_iters=5000, batch=64):
    # n_iters=5000：试过砍到 3000（fm_loss 看着已收敛到 0.016），但 **left 那一路
    # 会崩**（8 次采样平均的终点 y 只有 +8.7，真值 +19.96）。原因是聚合的 fm_loss
    # 是三个 mode 的平均，left 拖后腿时从曲线里看不出来。不要只凭 loss 判断收敛。
    model.train()
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        img = IMAGES[mode]
        r_in = REASONING_IN[mode]                         # (B,MAX) 用 <pad> 补齐
        r_lab = REASONING_LAB[mode]                       # (B,MAX) 用 IGNORE_INDEX 补齐

        # ① CosmosReason 训练：teacher forcing
        visual = model.vit(img).last_hidden_state         # (B,17,64)
        bos = torch.full((batch, 1), BOS_ID, dtype=torch.long)
        text_in = torch.cat([bos, r_in[:, :-1]], dim=1)        # (B,MAX)
        # cosmos 现在返回 (hidden, caches)：KV cache 版即使不增量生成也返回两份
        h, _ = model.cosmos(visual, text_in)
        text_h = h[:, VISUAL_TOKENS:]                     # (B,MAX)
        logits = model.cosmos.head(text_h)
        # ← 变长的代价：pad 位置用 ignore_index 跳过（同训练教程 train2 的标签掩码）
        cot_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            r_lab.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        # ② Expert 训练：condition = 推理前缀隐状态（去 <bos>），与推理时一致。
        # ★ 直接复用上面那次 forward 的 h —— 输入完全相同，不要再把 cosmos 跑一遍。
        #   两处用的是同一个 _condition_from_hidden，所以含义和推理时严格一致。
        model.condition = model._condition_from_hidden(h, text_in)
        x1 = target_actions[mode]
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(batch)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_target = x1 - x0
        v_pred = model.step_fn(x_t, t)
        fm_loss = F.mse_loss(v_pred, v_target)

        loss = cot_loss + fm_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"iter {it:4d}  cot_loss={cot_loss.item():.4f}  fm_loss={fm_loss.item():.4f}")


# ═══════════════════════════════════════════════════════════════
# Part 2：换一种「条件化」方式 —— prefix（Alpamayo 的真实做法）
# ═══════════════════════════════════════════════════════════════

class PrefixMiniVLA(nn.Module):
    """Part 2：prefix 方式（**Alpamayo 的真实做法**）。

    ★ 它【复用 Part 1 的 backbone】（ViT + CosmosReason），只换成 prefix 版 Expert：
      条件直接用 `backbone.generate()` 产出的 **cache** —— 里面【含 CoC】，
      而且因为 generate 已经是增量生成，**不需要再跑任何额外 forward**。

    这正是真实代码的结构：
        vlm_outputs = vlm.generate(...)
        prompt_cache = vlm_outputs.past_key_values
        expert(..., past_key_values=prompt_cache)
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone                       # 复用（含 vit + cosmos，且同层数）
        self.expert = CacheStack(n_layers=3)           # 和 cosmos 同层数 → 能接 cache
        self.in_proj = ActionInProj()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()

    def step_fn(self, x, t, caches, cache_pad_mask=None):
        """注意：**没有 condition 参数** —— 条件完全在 caches 里。"""
        emb = self.in_proj(x, t)
        h, _ = self.expert(emb, caches=caches, causal=False,
                           cache_pad_mask=cache_pad_mask)
        return self.out_proj(h)

    def sample(self, image):
        # ← 复用 Part 1 的生成（含 CoC）；pad 位置直接从前缀里屏蔽掉
        _, caches, cmask = self.backbone.generate(image)
        action = self.fm.sample(lambda x, t: self.step_fn(x, t, caches, cmask),
                                batch_size=image.shape[0])
        return self.action_space.action_to_traj(action)


def train_prefix(model, backbone, opt, target_actions, n_iters=5000, batch=64):
    """只训 prefix 版 Expert；backbone 冻结 → cache 【预计算一次】即可。

    这也是真实推理的做法：VLM 跑一次、cache 存下来，之后反复用。
    （早期版本每步都跑一次 generate，慢了 10 倍。）

    ★ 一个容易踩的坑：为了省时间而「每次迭代只训一个 mode」，会让一个 batch 里
      32 个样本的条件完全相同 —— 梯度等价于 1 个样本，收敛明显变差。
      正确做法是**把 3 个 mode 的 cache 预先堆成 (3,H,L,D)，每步按 mode 索引**，
      这样既有混合 batch，又不用重跑 generate。
    """
    model.train()
    # 三个 mode 【一次 batch 生成】：这样变长的三条链会一起跑到最长的那条，
    # cache 长度天然对齐（短的用 <pad> 补齐，靠 mask 屏蔽）。
    with torch.no_grad():
        text_ids_all, cs_all, masks_all = backbone.generate(IMAGES)
        # 逐层堆叠：(3, H, L, D)，之后用整数张量索引即可取到 (B, H, L, D)
        cache_bank = [(cs_all[li][0], cs_all[li][1]) for li in range(len(cs_all))]
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        caches = [(k[mode], v[mode]) for k, v in cache_bank]
        cmask = masks_all[mode]
        x1 = target_actions[mode]
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(batch)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        loss = F.mse_loss(model.step_fn(x_t, t, caches, cmask), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"  iter {it:4d}  loss={loss.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = KAPPA
    target[1, :, 1] = -KAPPA
    target[2, :, 1] = 0.0
    # 真值轨迹：注意 target 是【动作】（曲率），终点 y 要用 action_to_traj 积出来
    GT_TRAJ = ActionSpace().action_to_traj(target)

    print("三条推理链（长度【不同】）：")
    for i, c in enumerate(CHAINS):
        words = " ".join(VOCAB[t] for t in c)
        print(f"  {['left','right','straight'][i]:9s} {len(c)} 个 token:  {words}")
    print(f"  → 最长 {MAX_LEN}，训练时短的补齐、用 IGNORE_INDEX 不算 loss\n")

    model = MiniVLA()
    # lr=5e-4：left/right 推理链共享前缀，地形较陡，1e-3 在部分种子上会崩
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    train(model, opt, target)
    model.eval()

    print("\n训练后：给一张图，模型自回归【生成】推理（见到 <eos> 停）+ 预测轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj, text_ids = model.sample(IMAGES[i:i + 1])
            words = [VOCAB[t] for t in text_ids[0].tolist()]
            n_gen = len(text_ids[0]) - 1                  # 去掉 <bos>
            end = traj[0, -1, :2]
            print(f"  {name:9s} 生成 {n_gen} 个 token: {' '.join(words):<40} "
                  f"终点(x,y)=({end[0]:6.2f}, {end[1]:6.2f})  ← 单次采样，方差大")

    # ═══════════════════════════════════════════════════════════════
    # Part 2：换一种「条件化」方式（prefix —— Alpamayo 的真实做法）
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 2：同一个任务，改用 prefix 方式条件化")
    print("=" * 70)
    print("  Part 1 的调法:  expert(emb, condition)   ← 单独传条件张量")
    print("  Part 2 的调法:  expert(emb, caches)      ← 条件在 VLM 的逐层 K/V 里\n")

    # ★ 复用 Part 1 的 backbone（ViT + CosmosReason），只训一个新的 prefix Expert
    prefix_model = PrefixMiniVLA(backbone=model)
    opt2 = torch.optim.Adam(prefix_model.expert.parameters(), lr=5e-4)
    train_prefix(prefix_model, model, opt2, target)
    prefix_model.eval()

    print("\n  cache 结构（由 generate 增量产出，层数 = cosmos 的 block 数）：")
    with torch.no_grad():
        text_ids_p, cs, mask_p = model.generate(IMAGES)     # ← 三个 mode 一起生成
    for i, name in enumerate(names):
        words = " ".join(VOCAB[t] for t in text_ids_p[i].tolist())
        n_pad = int(mask_p[i].sum())
        print(f"    {name:9s} {words:<40} 前缀里的 <pad> 位置数 = {n_pad}")
    print(f"    前缀总长 = {cs[0][0].shape[2]}  （视觉 17 + <bos> 1 + 6）")
    for i, (k, v) in enumerate(cs):
        print(f"    layer {i}: K{tuple(k.shape)}  V{tuple(v.shape)}")
    print("    ↑ cache 里【含 CoC】—— 这才是真实 past_key_values 的内容")
    print("    ↑ 短的链（<pad> 位置）由 cache_pad_mask 在 attention 里屏蔽")

    # ★ 单次采样的终点 y 方差很大（同一个模型两次采样可能差 15m 以上），
    #   只报一个样本会得出「时好时坏」的假结论。采样本身很便宜（10 个小步），
    #   所以这里跑 N_SAMPLES 次取平均再判类。
    N_SAMPLES = 8
    print(f"\n  两种方式的输出对比（终点 y，各取 {N_SAMPLES} 次采样平均；"
          f"真值来自把 target 动作积分成轨迹）：")
    print(f"  {'':<10}{'① cross-attn':>18}{'② prefix':>18}{'真值':>10}")
    gt_all = GT_TRAJ[:, -1, 1]                          # (3,) 三个真值
    # straight 的真值是 0，用「符号相同」判对错是没有意义的。
    # 改成分类口径：预测的 y 最接近哪个真值，就算认成哪一类。
    def nearest_gt(y):
        return int((gt_all - y).abs().argmin())
    n_ok = 0
    with torch.no_grad():
        for i, name in enumerate(names):
            ya = torch.stack([model.sample(IMAGES[i:i + 1])[0][0, -1, 1]
                              for _ in range(N_SAMPLES)]).mean().item()
            yb = torch.stack([prefix_model.sample(IMAGES[i:i + 1])[0, -1, 1]
                              for _ in range(N_SAMPLES)]).mean().item()
            ma, mb = nearest_gt(ya) == i, nearest_gt(yb) == i
            print(f"  {name:<10}{ya:>+14.2f}{'✓' if ma else '✗':>3}"
                  f"{yb:>+16.2f}{'✓' if mb else '✗':>3}{gt_all[i]:>+10.2f}")
            n_ok += int(ma) + int(mb)
    print(f"\n  → 6 个预测里 {n_ok} 个分类正确（3 个 mode × 2 种方式）")
    print("  → 两种方式用的是【同一个 backbone、同一份条件】，差别只在连接方式：")
    print("     cross-attn 传 hidden 张量，prefix 传逐层 K/V")
