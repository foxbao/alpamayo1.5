"""Stage 14: 完整输入 + CoC —— 历史 + 图片一起进 CosmosReason

【目的】把「H（历史）+ V（视觉）+ L（推理）+ A（动作）」串成 toy 里最完整的输入侧：
  - 对比 stage13：前缀从【只有视觉】扩展成【历史 + 视觉】多模态前缀
  - 融合方式：历史经 HistoryEncoder → (B,8,64)，与视觉 (B,17,64) 在序列维 concat，
    再一起喂给 CosmosReason 生成 CoC
  - **前缀是可以任意长、任意模态的**——这正是真实 `past_key_values` 的性质：
    「动作 token 之前的所有内容」都装在同一条序列的逐层 K/V 里
  - 训练同样是 cot_loss + fm_loss 联合，pad 处理同 stage13（ignore_index + attention mask）

★ 和 stage13 一样分两部分，**Part 1 是对照组、Part 2 是真实做法**（详见 stage13 的说明）：
  - Part 1【对照组】：cross-attn 条件化，toy 主干 stage4~12 的做法
  - Part 2【真实做法】：prefix 续写，generate 用 KV cache 增量生成，
    prefix 版 Expert 直接复用那份 cache —— 区别只是**前缀是多模态的**（历史 + 视觉）。
    cache 长度 = 历史 8 + 视觉 17 + <bos> 1 + 生成 6。
  两者是【对等的连接拓扑】，不是「进阶关系」；选 Part 2 的理由是真实代码就那么写的。

【简化】
  - 历史只有 8 步位移增量 (dx,dy)，且是**合成的**（与图片同方向解析生成）；
    真实是 48 个 <|traj_history|> 占位符（16 位姿 × 3 维 xyz），由 tokenizer 编码
  - 只有 1 个相机、1 帧；真实是多相机（4~8 路）× 每路 4 帧 + 相机文字标签
  - 没有导航指令段，没有 <|route_start|>...<|route_end|> 结构，
    也没有 <|cot_start|>/<|image_start|> 等 special token 框架
  - CosmosReason 3 层 / hidden 64 / 随机初始化；真实 Cosmos-Reason2-8B（30+ 层、预训练）
  - 词表 11 个 token、3 条人工推理链；真实约 15 万词表
  - 历史进入模型的路径仍然只有「VLM 条件」一条；真实里历史还同时用于
    `estimate_t0_states` 估计积分初速度 v0（见 real6 的【隔离】实验）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
    build_vit, CosmosReason, HistoryEncoder, CacheStack,
)

KAPPA = 0.1
IMG_SIZE = 16
HIST_LEN = 8          # 历史步数
VISUAL_TOKENS = 17    # ViT 输出：1 CLS + 16 patch

# CoC 词表 + 推理链（同 stage13）
# ⚠️ 这些是模型【生成】的推理，不是输入指令。
#    对比 stage10 的【输入】指令词：turn / left / right / keep / straight / in / 10m / 30m
VOCAB = ["<bos>", "<eos>", "<pad>",
         "shift", "left", "right", "hold", "course", "due", "to", "curve"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BOS_ID, EOS_ID, PAD_ID = V["<bos>"], V["<eos>"], V["<pad>"]
IGNORE_INDEX = -100

# 三条推理链，长度【不同】（真实 CoC 是变长的）
CHAINS = [
    [V["shift"], V["left"], V["due"], V["to"], V["curve"], EOS_ID],    # 6 个
    [V["shift"], V["right"], V["due"], V["to"], V["curve"], EOS_ID],   # 6 个
    [V["hold"], V["course"], EOS_ID],                                  # 3 个 ← 更短
]
MAX_LEN = max(len(c) for c in CHAINS)

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


def make_history(mode):
    """历史轨迹：和图片同方向的一段位移增量序列。"""
    kappa = [KAPPA, -KAPPA, 0.0][mode]
    theta = 0.0
    ds = []
    for _ in range(HIST_LEN):
        theta += kappa * 5.0 * 0.1
        ds.append([5.0 * math.cos(theta) * 0.1, 5.0 * math.sin(theta) * 0.1])
    return torch.tensor(ds)   # (HIST_LEN, 2)


IMAGES = torch.stack([make_image(0), make_image(1), make_image(2)])        # (3,1,16,16)
HISTORIES = torch.stack([make_history(0), make_history(1), make_history(2)])  # (3,HIST_LEN,2)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_enc = HistoryEncoder()
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

    def cache_pad_mask(self, text_ids):
        """前缀里哪些位置是 <pad>（变长生成补出来的）→ 交给 Expert 屏蔽。

        前缀 = [历史 HIST_LEN | 视觉 17 | 文本 ...]，只有文本部分可能有 <pad>。
        """
        B = text_ids.shape[0]
        return torch.cat([torch.zeros(B, HIST_LEN + VISUAL_TOKENS, dtype=torch.bool),
                          text_ids == PAD_ID], dim=1)

    def generate(self, hist, img, max_len=MAX_LEN):
        """历史 + 图片 → 融合 → 自回归生成 CoC（**增量 + KV cache**，见到 <eos> 停）。

        返回 `(text_ids, caches, cache_pad_mask)`：caches 是
        [历史 + 视觉 + <bos> + 生成内容] 的逐层 K/V —— 就是真实 VLM `generate()`
        留下的 `past_key_values`，Part 2 的 prefix Expert 直接拿它当条件
        （不用再跑第二次 forward）。
        """
        B = hist.shape[0]
        hist_embeds = self.hist_enc(hist)                    # (B, HIST_LEN, 64)
        visual = self.vit(img).last_hidden_state             # (B, 17, 64)
        input_embeds = torch.cat([hist_embeds, visual], dim=1)  # (B, HIST_LEN+17, 64) 融合
        self._input_embeds = input_embeds                    # 供 Part 1 取隐状态用

        text_ids = torch.full((B, 1), BOS_ID, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool)          # 逐个样本记录"是否已吐 eos"
        # prefill：整条前缀（历史 + 视觉 + <bos>）一次算完，留下逐层 K/V
        h, caches = self.cosmos(input_embeds, text_ids)
        for _ in range(max_len):
            next_tok = self.cosmos.head(h[:, -1]).argmax(-1, keepdim=True)
            # 已结束的样本继续喂 <pad>（不能再让它生成，否则会污染）
            next_tok = torch.where(
                finished[:, None], torch.full_like(next_tok, PAD_ID), next_tok
            )
            text_ids = torch.cat([text_ids, next_tok], dim=1)
            finished |= (next_tok.squeeze(-1) == EOS_ID)
            # ★ 增量：只喂新 token，复用 cache —— 不重算整条前缀
            h, caches = self.cosmos(input_embeds, next_tok, caches)
            if bool(finished.all()):
                break                                        # ← 变长：全 batch 都停了才退出
        return text_ids, caches, self.cache_pad_mask(text_ids)

    def _condition_from_hidden(self, h, input_embeds, text_in):
        """从【已经算好的】h 里取出推理隐状态（去掉 <bos>）并标记 <pad> 位置。"""
        self.cond_pad_mask = text_in[:, 1:] == PAD_ID         # (B, L-1) True=pad
        return h[:, input_embeds.shape[1] + 1 :]             # 去掉 <bos>

    def _reasoning_hidden(self, input_embeds, text_in):
        """给定 [<bos>, r1, r2, ...]，返回推理隐状态（去掉 <bos>）并标记 <pad> 位置。

        只有 Part 1（cross-attn 版）需要它 —— 因为 cross-attention 要的是 hidden
        张量，不是 K/V。Part 2 直接用 generate 产出的 cache。
        """
        h, _ = self.cosmos(input_embeds, text_in)
        return self._condition_from_hidden(h, input_embeds, text_in)

    def sample(self, hist, img):
        text_ids, _, _ = self.generate(hist, img)
        B = hist.shape[0]
        # 把生成结果补齐到定长，使 condition 的形状与训练一致
        gen = text_ids[:, 1:]                                # 去掉 <bos>
        if gen.shape[1] < MAX_LEN:
            pad = torch.full((B, MAX_LEN - gen.shape[1]), PAD_ID, dtype=torch.long)
            gen = torch.cat([gen, pad], dim=1)
        text_in = torch.cat([torch.full((B, 1), BOS_ID, dtype=torch.long), gen], dim=1)
        self.condition = self._reasoning_hidden(self._input_embeds, text_in)
        action = self.fm.sample(self.step_fn, batch_size=B)
        traj = self.action_space.action_to_traj(action)
        return traj, text_ids


def train(model, opt, target_actions, n_iters=5000, batch=64):
    # n_iters=5000：同 stage13——砍到 3000 时聚合 fm_loss 看着收敛了，
    # 但 left 那一路实际会崩。不要只凭 loss 判断收敛。
    model.train()
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        hist = HISTORIES[mode]
        img = IMAGES[mode]
        r_in = REASONING_IN[mode]                         # (B,MAX) 用 <pad> 补齐
        r_lab = REASONING_LAB[mode]                       # (B,MAX) 用 IGNORE_INDEX 补齐

        # ① CosmosReason：teacher forcing
        hist_embeds = model.hist_enc(hist)                   # (B,H,64)
        visual = model.vit(img).last_hidden_state            # (B,17,64)
        input_embeds = torch.cat([hist_embeds, visual], dim=1)
        bos = torch.full((batch, 1), BOS_ID, dtype=torch.long)
        text_in = torch.cat([bos, r_in[:, :-1]], dim=1)      # (B,MAX)
        h, _ = model.cosmos(input_embeds, text_in)
        text_h = h[:, input_embeds.shape[1]:]                # 用于语言建模损失
        logits = model.cosmos.head(text_h)
        # ← 变长的代价：pad 位置用 ignore_index 跳过（同训练教程 train2 的标签掩码）
        cot_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            r_lab.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        # ② Expert：condition = 推理前缀隐状态（去 <bos>），与推理时一致。
        # ★ 直接复用上面那次 forward 的 h —— 输入完全相同，不要再把 cosmos 跑一遍。
        model.condition = model._condition_from_hidden(h, input_embeds, text_in)
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
    """同 stage13 的 Part 2，但**前缀是多模态的**：历史 token + 视觉 token。

    这正是真实里 `past_key_values` 装的东西 —— 「动作之前的所有内容」，
    可以包含任意长度、任意模态。注意 `step_fn` 里【没有 condition 参数】。

    ★ 它【复用 Part 1 的 backbone】（HistoryEncoder + ViT + CosmosReason），
      条件直接用 `backbone.generate()` 产出的 cache —— 里面含 CoC。
      cache 长度 = 历史 8 + 视觉 17 + <bos> 1 + 生成 6 —— 多模态就这么被「拼进一条序列」。
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone                       # 复用（含 hist_enc + vit + cosmos）
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

    def sample(self, hist, img):
        # ← 复用 Part 1 的生成（含 CoC）；pad 位置直接从前缀里屏蔽掉
        _, caches, cmask = self.backbone.generate(hist, img)
        action = self.fm.sample(lambda x, t: self.step_fn(x, t, caches, cmask),
                                batch_size=hist.shape[0])
        return self.action_space.action_to_traj(action)


def train_prefix(model, backbone, opt, target_actions, n_iters=5000, batch=64):
    """只训 prefix 版 Expert；backbone 冻结 → cache 【预计算一次】即可。

    ★ 同 stage13：不要「每次迭代只训一个 mode」（那样一个 batch 里条件全同，
      梯度等价于 1 个样本）。把 3 个 mode 的 cache 堆成 (3,H,L,D) 再按 mode 索引。
    """
    model.train()
    # 三个 mode 【一次 batch 生成】：变长的链一起跑到最长的那条，cache 长度天然对齐
    with torch.no_grad():
        text_ids_all, cs_all, masks_all = backbone.generate(HISTORIES, IMAGES)
        cache_bank = [cs_all[li] for li in range(len(cs_all))]   # 每层 (3, H, L, D)
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
    # 真值轨迹：target 是【动作】（曲率），终点 y 要用 action_to_traj 积出来
    GT_TRAJ = ActionSpace().action_to_traj(target)

    # ═══════════════════════════════════════════════════════════════
    # Part 1：【对照组】—— toy 主干一直用的 cross-attention 条件化
    # ═══════════════════════════════════════════════════════════════
    # 先跑这条是为了给 Part 2 一个可比较的基准：同一个 backbone、同一份数据、
    # 同一个任务，唯一变量是「条件怎么连到 Expert」（详见 stage13 的说明）。
    print("=" * 70)
    print("Part 1【对照组】：cross-attention 条件化（toy 主干 stage4~12 的做法）")
    print("=" * 70)
    print("  条件是一份【显式的 hidden 张量】(B, T, 64)，用 expert(emb, condition) 传入")
    print("  → 好处：能直接 print(condition.shape) 看清条件里有什么（stage7~12 就靠这个）")
    print("  → 代价：cross-attn 要 hidden 而非 K/V，所以推理时必须【再跑一次】完整 forward\n")

    model = MiniVLA()
    # lr=5e-4：同 stage13，left/right 推理链共享前缀，地形更陡，1e-3 不稳定
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    train(model, opt, target)
    model.eval()

    print("三条推理链（长度【不同】）：")
    for i, c in enumerate(CHAINS):
        print(f"  {['left','right','straight'][i]:9s} {len(c)} 个 token:  {' '.join(VOCAB[t] for t in c)}")

    print("\n  对照组的输出：历史 + 图片一起给，模型自回归【生成】推理（见到 <eos> 停）+ 预测轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj, text_ids = model.sample(HISTORIES[i:i + 1], IMAGES[i:i + 1])
            words = [VOCAB[t] for t in text_ids[0].tolist()]
            n_gen = len(text_ids[0]) - 1                  # 去掉 <bos>
            end = traj[0, -1, :2]
            print(f"  {name:9s} 生成 {n_gen} 个 token: {' '.join(words):<40} "
                  f"终点(x,y)=({end[0]:6.2f}, {end[1]:6.2f})  ← 单次采样，方差大")

    # ═══════════════════════════════════════════════════════════════
    # Part 2：【Alpamayo 的真实做法】—— prefix 续写（前缀是多模态的）
    # ═══════════════════════════════════════════════════════════════
    # 和 Part 1 是【两种对等的连接拓扑】，不是「进阶版」。
    print("\n" + "=" * 70)
    print("Part 2【真实做法】：prefix 续写（条件 = VLM 留下的逐层 K/V）")
    print("=" * 70)
    print("  Part 1:  expert(emb, condition)   ← 单独传条件张量")
    print("  Part 2:  expert(emb, caches)      ← 条件在 VLM 的逐层 K/V 里")
    print("           注意 step_fn 签名里【没有 condition 参数】")
    print("  ★ 前缀 = 历史 token + 视觉 token —— 多个模态拼成一条序列\n")

    # ★ 复用 Part 1 的 backbone，只训一个新的 prefix Expert
    prefix_model = PrefixMiniVLA(backbone=model)
    opt2 = torch.optim.Adam(prefix_model.expert.parameters(), lr=5e-4)
    train_prefix(prefix_model, model, opt2, target)
    prefix_model.eval()

    print("\n  前缀结构（这就是真实 past_key_values 装的东西）：")
    with torch.no_grad():
        text_ids_p, cs, mask_p = model.generate(HISTORIES, IMAGES)
    for i, name in enumerate(names):
        words = " ".join(VOCAB[t] for t in text_ids_p[i].tolist())
        print(f"    {name:9s} {words:<40} 前缀里的 <pad> 位置数 = {int(mask_p[i].sum())}")
    print(f"    step1 编码前缀：历史 {HIST_LEN} + 视觉 {VISUAL_TOKENS} = {HIST_LEN + VISUAL_TOKENS} 个 token")
    print(f"    step2 自回归生成 6 个 token（含 <eos>）")
    print(f"    → cache 总长 = {cs[0][0].shape[2]}")
    for i, (k, v) in enumerate(cs):
        print(f"    layer {i}: K{tuple(k.shape)}  V{tuple(v.shape)}")
    print("    ↑ 前缀是【多模态】的：历史 + 视觉 + CoC 全在同一条序列的 K/V 里")

    # ★ 单次采样的终点 y 方差很大（同一个模型两次采样可能差 15m 以上），
    #   只报一个样本会得出「时好时坏」的假结论。采样本身很便宜（10 个小步），
    #   所以这里跑 N_SAMPLES 次取平均再判类。
    N_SAMPLES = 8
    print(f"\n  对照结果（终点 y，各取 {N_SAMPLES} 次采样平均；"
          f"真值来自把 target 动作积分成轨迹）：")
    print(f"  {'':<10}{'① cross-attn(对照)':>20}{'② prefix(真实)':>18}{'真值':>10}")
    gt_all = GT_TRAJ[:, -1, 1]                          # (3,) 三个真值
    # straight 的真值是 0，用「符号相同」判对错是没有意义的。
    # 改成分类口径：预测的 y 最接近哪个真值，就算认成哪一类。
    def nearest_gt(y):
        return int((gt_all - y).abs().argmin())
    n_ok = 0
    with torch.no_grad():
        for i, name in enumerate(names):
            ya = torch.stack([model.sample(HISTORIES[i:i + 1], IMAGES[i:i + 1])[0][0, -1, 1]
                              for _ in range(N_SAMPLES)]).mean().item()
            yb = torch.stack([prefix_model.sample(HISTORIES[i:i + 1], IMAGES[i:i + 1])[0, -1, 1]
                              for _ in range(N_SAMPLES)]).mean().item()
            ma, mb = nearest_gt(ya) == i, nearest_gt(yb) == i
            print(f"  {name:<10}{ya:>+14.2f}{'✓' if ma else '✗':>3}"
                  f"{yb:>+16.2f}{'✓' if mb else '✗':>3}{gt_all[i]:>+10.2f}")
            n_ok += int(ma) + int(mb)
    print(f"\n  → 6 个预测里 {n_ok} 个分类正确（3 个 mode × 2 种方式）")
    print("  → 同一个 backbone、同一份条件，只是连接方式不同：")
    print("     cross-attn 传 hidden 张量，prefix 传逐层 K/V")
    print("  → 这里要看的不是分数高低，而是【两种连接拓扑是否都能把条件传到动作】。")
    print("     两边都对上 → 说明条件传递并不依赖 cross-attn。")
    print("  → 但这不是「②比①更好」：单一 seed、这么小的模型不构成效果比较；")
    print("     选 prefix 的理由是【真实代码就那么写的】，不是它在 toy 上跑分更高。")
