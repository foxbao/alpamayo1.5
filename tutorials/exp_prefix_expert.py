"""exp: Prefix 版 Expert —— Alpamayo 真实用的「条件化」方式

对比 toy 的 cross-attention 版（stage4~15 用的）：

    toy（两路 · cross-attention）:
        动作 token ──► [Expert] ──┐
                                  ├─ cross-attn 连接两路
        条件 (B,L,64) ────────────┘

    真实（一路 · prefix 续写）:
        VLM 的 KV cache（前缀）┐
                               ├─► 拼成【一条序列】──► [Expert] ──► 输出
        动作 token ────────────┘

实测依据（`Qwen3VLTextModel` 第 0 层）：真实 Expert 只有
`self_attn + mlp + 2×RMSNorm`——**没有 cross-attention**，和 VLM 文本塔结构完全相同。

本实验：用同一个架构实现 prefix 版，并训练到能收敛（任务同 stage6）。
纯 CPU，几十秒。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ActionSpace, FlowMatching, N_WAYPOINTS, ACTION_DIM, HIDDEN, DT

D = HIDDEN               # 64
N_LAYERS = 3             # 层数：VLM 和 Expert 必须一致，否则 cache 不能共用
PREFIX_LEN = 2           # 前缀长度（真实里是几千）
N_COND = 3               # 三个条件
KAPPA = 0.05


class CacheBlock(nn.Module):
    """因果 transformer block（支持 KV cache）。VLM 和 Expert 共用同一种 block。"""

    def __init__(self, d=D, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None, causal=True):
        """x: (B, T, d)；cache: (k_prev, v_prev)；causal=False 时动作之间全可见。"""
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # 拆成多头 (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))

        if cache is not None:                      # ← 拼接 VLM 缓存的前缀 K/V
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        attn = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if causal:                                 # 前缀内部是因果的
            L = x.shape[1]
            attn = attn + torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        # causal=False（Expert）：动作 token 可以看到 前缀 + 全部动作，不加 mask

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        x = x + self.proj(out)
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class CacheStack(nn.Module):
    """N 层 CacheBlock 叠起来。VLM 和 Expert 都用它。"""

    def __init__(self, d=D, n_layers=N_LAYERS):
        super().__init__()
        self.blocks = nn.ModuleList([CacheBlock(d) for _ in range(n_layers)])

    def forward(self, x, caches=None, causal=True):
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i], causal=causal)
            new_caches.append(c)
        return x, new_caches


class PrefixMiniVLA(nn.Module):
    """prefix 版：VLM 编码前缀 → 产 cache；Expert 接在后面算动作。"""

    def __init__(self):
        super().__init__()
        # VLM：把"条件"编码成前缀（真实里是 图+历史+文本 → token）
        self.cond_embed = nn.Embedding(N_COND, D)
        self.prefix_pos = nn.Parameter(torch.zeros(1, PREFIX_LEN, D))
        self.vlm = CacheStack()
        # Expert：**同一种 CacheBlock、同样的层数** —— 所以能接 VLM 的 cache
        self.expert = CacheStack()
        # 动作 token 的投影（toy 里是 ActionInProj，这里简化）
        self.action_in = nn.Linear(ACTION_DIM, D)
        self.out_proj = nn.Linear(D, ACTION_DIM)

    def build_prefix(self, cond_idx):
        """把条件变成前缀 token。同一个 cond 的两个 token 都带条件信息。"""
        B = cond_idx.shape[0]
        c = self.cond_embed(cond_idx)[:, None, :]              # (B,1,D)
        return c.expand(B, PREFIX_LEN, -1) + self.prefix_pos

    def step_fn(self, x, t, caches):
        """扩散每一步：动作 token 接在前缀 cache 后面算。"""
        B, T, _ = x.shape
        emb = self.action_in(x)                                # (B,64,D)
        h, _ = self.expert(emb, caches=caches, causal=False)   # ← 复用 VLM 的 cache
        return self.out_proj(h)

    def sample(self, cond_idx, fm, action_space):
        prefix = self.build_prefix(cond_idx)
        _, caches = self.vlm(prefix)                           # ← VLM 跑一次，留下 cache
        action = fm.sample(lambda x, t: self.step_fn(x, t, caches),
                           batch_size=cond_idx.shape[0])
        return action_space.action_to_traj(action)


def train(model, opt, target_actions, fm, n_iters=2000, batch=32):
    model.train()
    for it in range(n_iters):
        idx = torch.randint(0, N_COND, (batch,))
        prefix = model.build_prefix(idx)
        _, caches = model.vlm(prefix)                          # prefill：一次
        x1 = target_actions[idx]
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(batch)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_pred = model.step_fn(x_t, t, caches)
        loss = F.mse_loss(v_pred, x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 400 == 0:
            print(f"iter {it:4d}  loss={loss.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    fm, action_space = FlowMatching(), ActionSpace()
    target = torch.zeros(N_COND, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = KAPPA
    target[1, :, 1] = -KAPPA
    target[2, :, 1] = 0.0

    print("=" * 70)
    print("结构：VLM 与 Expert 必须【同一种 CacheBlock、同层数】才能共用 cache")
    print("=" * 70)
    print(f"  VLM:    {N_LAYERS} 层 CacheBlock   （条件 → 前缀 → 产 cache）")
    print(f"  Expert: {N_LAYERS} 层 CacheBlock   （动作 token 接在前缀后面）")
    print(f"  前缀长度 = {PREFIX_LEN}（真实里是几千）\n")

    model = PrefixMiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    train(model, opt, target, fm)

    # ---- 展示 cache 的层结构（这就是「层」的具象化）----
    print("\n" + "=" * 70)
    print("VLM 产出的 cache —— 每一层都有一份 K/V")
    print("=" * 70)
    with torch.no_grad():
        _, caches = model.vlm(model.build_prefix(torch.tensor([0])))
    print(f"  层数 = {len(caches)}（= CacheBlock 的个数）")
    for i, (k, v) in enumerate(caches):
        print(f"  layer {i}:  K{tuple(k.shape)}  V{tuple(v.shape)}   "
              f"（B, heads, 前缀长度, head_dim）")

    # ---- 结果 ----
    print("\n" + "=" * 70)
    print("训练后：给条件 → 采样轨迹（动作 token 完全靠 cache 拿到条件）")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        for name, i in [("left", 0), ("right", 1), ("straight", 2)]:
            traj = model.sample(torch.tensor([i]), fm, action_space)
            end = traj[0, -1, :2]
            print(f"  {name:9s} 终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")

    print("\n" + "=" * 70)
    print("对比 toy 的 cross-attention 版")
    print("=" * 70)
    print("  toy:    动作一路、条件一路，Expert 里多一个 cross_attn 模块")
    print(f"          条件必须【单独传】(B,L,{D})")
    print("  本实验: 只有 self_attn，动作 token「续写」在前缀后面")
    print("          → 条件通过 VLM 的【逐层 K/V】传递，不需要单独的 condition 张量")
    print("          → 所以真实代码里能看到 expert(past_key_values=prompt_cache)")
