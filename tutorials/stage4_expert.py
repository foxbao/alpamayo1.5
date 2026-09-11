"""Stage 4: 条件化 —— 两种做法，同一个目标

目标：让「动作」依赖「条件」。有两种实现方式，本 stage 两个都写、都跑、对比：

    ① cross-attention（通用做法）
       动作 token ──► [Expert] ──┐
                                 ├─ cross_attn 连接两路
       条件 (B,L,H) ─────────────┘
       → 广泛使用：Stable Diffusion 的文本条件、原始 Transformer 的 decoder

    ② prefix 续写（**Alpamayo 的真实做法**）
       VLM 的逐层 K/V（cache）┐
                              ├─► 拼成【一条序列】──► [Expert] ──► 输出
       动作 token ────────────┘
       → Expert 和 VLM 文本塔【结构完全相同】，没有 cross-attention
       → 条件通过 cache 传递，没有单独的 condition 参数

实测依据：真实 Expert 的第 0 层是 `self_attn + mlp + 2×RMSNorm`，
和 VLM 文本塔的第 0 层一模一样，看不到任何 cross-attention 模块。

详见 `exp_prefix_expert.py`（把 ② 训练到收敛）和 README §六。
"""

import math

import torch
import torch.nn as nn

BATCH = 2
N_WAYPOINTS = 64
HIDDEN = 64
VLM_SEQ_LEN = 32
N_HEADS = 4
PREFIX_LEN = 4          # ② 里的前缀长度（真实里是几千）


# ═══════════════════════════════════════════════════════════════
# 做法 ①：cross-attention（通用做法）
# ═══════════════════════════════════════════════════════════════

class CrossAttnBlock(nn.Module):
    """一个 transformer block：self-attn → cross-attn → FFN。"""

    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x, condition):
        # ① self-attention：动作 token 之间互相看（非因果！和 LLM 相反）
        x = x + self.self_attn(x, x, x)[0]
        x = self.norm1(x)
        # ② cross-attention：动作当 query，去"查" 条件
        x = x + self.cross_attn(query=x, key=condition, value=condition)[0]
        x = self.norm2(x)
        # ③ FFN
        x = x + self.ffn(x)
        x = self.norm3(x)
        return x


class CrossAttnExpert(nn.Module):
    def __init__(self, hidden=HIDDEN, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttnBlock(hidden) for _ in range(n_blocks)])

    def forward(self, x, condition):
        for blk in self.blocks:
            x = blk(x, condition)
        return x


# ═══════════════════════════════════════════════════════════════
# 做法 ②：prefix 续写（Alpamayo 的真实做法）
# ═══════════════════════════════════════════════════════════════

class Block(nn.Module):
    """因果 transformer block，**支持 KV cache**。

    注意它和 ① 的 ExpertBlockCrossAttn 的区别：**没有 cross_attn**。
    它和 VLM 用的 block 是同一种东西——这正是两者能共用 cache 的前提。
    """

    def __init__(self, hidden=HIDDEN, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x, cache=None, causal=True):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))

        if cache is not None:                       # ← 拼接前缀的 K/V（来自 VLM）
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        attn = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if causal:                                  # 前缀内部是因果的
            attn = attn + torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        # causal=False（Expert）：动作 token 可以看到 前缀 + 全部动作
        attn = torch.softmax(attn, dim=-1)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, T, -1))
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class PrefixVLM(nn.Module):
    """把「条件」编码成前缀 token，并产出逐层 K/V（cache）。

    真实里这一步是 VLM 干的事：图 + 历史 + 文本 → 前缀 → 生成时留下 cache。
    """

    def __init__(self, hidden=HIDDEN, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(hidden, hidden)      # 假装把条件投影成前缀 token
        self.pos = nn.Parameter(torch.zeros(1, PREFIX_LEN, hidden))
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(n_layers)])

    def forward(self, condition_vec):
        # condition_vec: (B, H) —— 简化的「条件」；这里直接扩成 PREFIX_LEN 个 token
        x = self.embed(condition_vec)[:, None, :].expand(-1, PREFIX_LEN, -1) + self.pos
        caches = []
        for blk in self.blocks:
            x, c = blk(x, None, causal=True)        # 前缀内部：因果
            caches.append(c)
        return caches                               # ← 逐层 K/V


class PrefixExpert(nn.Module):
    """接在 VLM 的 cache 后面算动作。**注意 forward 里没有 condition 参数。**"""

    def __init__(self, hidden=HIDDEN, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(n_layers)])

    def forward(self, x, caches):
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x, cache=caches[i], causal=False)   # ← 条件从 cache 里来
        return x


# ═══════════════════════════════════════════════════════════════
# 对比
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    action_embeds = torch.randn(BATCH, N_WAYPOINTS, HIDDEN)     # 动作（stage3 的产物）
    condition = torch.randn(BATCH, VLM_SEQ_LEN, HIDDEN)         # 条件（32 个 token）

    print("=" * 70)
    print("① cross-attention（通用做法）")
    print("=" * 70)
    expert_a = CrossAttnExpert()
    out_a = expert_a(action_embeds, condition)
    print(f"  动作 embedding: {tuple(action_embeds.shape)}")
    print(f"  条件:           {tuple(condition.shape)}    ← 【单独传进去的】")
    print(f"  输出:           {tuple(out_a.shape)}")
    out_a0 = expert_a(action_embeds, torch.zeros_like(condition))
    print(f"  条件=随机 vs 条件=全0 的差异: {(out_a - out_a0).abs().mean():.4f}"
          f"   （>0 → cross-attn 确实在读条件）")

    print("\n" + "=" * 70)
    print("② prefix 续写（Alpamayo 的真实做法）")
    print("=" * 70)
    vlm, expert_b = PrefixVLM(), PrefixExpert()
    cond_vec = torch.randn(BATCH, HIDDEN)                       # 简化的「条件」
    caches = vlm(cond_vec)                                      # VLM 跑一次 → 逐层 cache
    out_b = expert_b(action_embeds, caches)
    print(f"  VLM 产出 cache: {len(caches)} 层")
    for i, (k, v) in enumerate(caches):
        print(f"    layer {i}: K{tuple(k.shape)}  V{tuple(v.shape)}")
    print(f"  动作 embedding: {tuple(action_embeds.shape)}")
    print(f"  输出:           {tuple(out_b.shape)}")
    print(f"  ⚠️ 注意：expert_b(action_embeds, caches) —— 【没有 condition 参数】")
    caches2 = vlm(torch.randn(BATCH, HIDDEN))                    # 换个条件
    out_b2 = expert_b(action_embeds, caches2)
    print(f"  换个条件后输出的差异: {(out_b - out_b2).abs().mean():.4f}"
          f"   （>0 → 条件确实从 cache 传进来了）")

    print("\n" + "=" * 70)
    print("对比")
    print("=" * 70)
    print(f"  {'':<14}{'① cross-attention':<26}{'② prefix（真实）':<26}")
    print(f"  {'数据流':<12}{'两路（动作 / 条件）':<24}{'一路（动作续写在前缀后）':<24}")
    print(f"  {'连接方式':<12}{'cross_attn 模块':<25}{'self_attn 看前缀':<25}")
    print(f"  {'条件怎么传':<11}{'单独传 (B,L,H)':<25}{'VLM 的逐层 K/V':<25}")
    print(f"  {'Expert 结构':<12}{'self+cross+FFN':<26}{'和 VLM 完全相同':<24}")
    print(f"  {'能共用 cache':<12}{'✗ 不能':<27}{'✓ 能（层数/结构对齐即可）':<20}")
