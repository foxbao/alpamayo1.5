"""exp: KV cache —— 自回归生成时，为什么不用重算前面所有 token？（读 stage13 之后做）

【目的】把「为什么能复用 cache」这件事亲手验一遍，而不是只信结论：
  用一个极小的因果语言模型对比两种生成方式：
    A) 每步重算整条前缀              —— 直觉做法（stage13 的早期版本）
    B) 每步只算新 token + 复用 cache —— 真实 LLM 推理的做法
  要证明的两件事：
    - **输出完全相同**（浮点误差内）—— 缓存不改变结果，只改变代价
    - **计算量差很多**：实测序列长度 4→512 时，两者差距从 7 倍拉到 384 倍（平方效应）
  这正是真实代码的结构：
      # alpamayo1_5.py:304
      prompt_cache = vlm_outputs.past_key_values    # ← 生成时留下的 KV cache
      expert(..., past_key_values=prompt_cache)     # ← 直接复用，不再重算

【简化】
  - 用一个**极小的合成因果 LM**（D_MODEL=64、2 层、词表 8）演示原理，
    不是 release 模型测速；倍数只反映 token 处理量的平方增长
  - 不涉及视觉 token、多相机、变长 batch，也没有真实 attention kernel 的开销差异
  - stage13 的 `generate` 已经是增量版（同本实验的 B）；本实验的价值在于
    把两种实现的**输出一致性**和**成本差距**都量化出来

【前提】纯 CPU，几秒。不属于 stage 主线，默认不纳入回归测试。
"""

import math
import time

import torch
import torch.nn as nn

D_MODEL = 64
N_LAYERS = 2
VOCAB_SIZE = 8
MAX_LEN = 16


class Block(nn.Module):
    """一个因果 transformer block，支持可选 KV cache。"""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)          # 一次算出 q/k/v
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        """x: (B, T, d)；cache: (k_prev, v_prev)，形状 (B, L, d)。

        用了 cache 时 T=1（只喂新 token）；否则 T=整条前缀长度。
        """
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)          # 各 (B, T, d)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)          # ← 关键：拼接缓存的 K
            v = torch.cat([cache[1], v], dim=1)          # ← 拼接缓存的 V
        new_cache = (k, v)

        attn = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])   # (B, T, L)
        if cache is None:                                # 没缓存 → 需要因果 mask
            L = x.shape[1]
            mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)

        x = x + self.proj(attn @ v)
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, ids, caches=None):
        """ids: (B, T)。caches: 每层的 (k,v)，None 表示不用缓存。"""
        x = self.embed(ids)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i])
            new_caches.append(c)
        return self.head(x), new_caches


@torch.no_grad()
def generate_naive(model, prompt, n_new):
    """A) 每步重算整条前缀（stage13 现在的做法）。返回 (生成的 ids, 处理过的 token 总数)。"""
    ids = prompt
    tokens_processed = 0
    for _ in range(n_new):
        logits, _ = model(ids)                    # ← 每次都跑整条前缀
        tokens_processed += ids.shape[1]
        nxt = logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids, tokens_processed


@torch.no_grad()
def generate_cached(model, prompt, n_new):
    """B) prefill 一次 + 每步只算新 token（真实 LLM 推理的做法）。"""
    logits, caches = model(prompt)                # ← prefill：整条前缀跑一次
    tokens_processed = prompt.shape[1]
    ids = prompt
    for _ in range(n_new):
        nxt = logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
        logits, caches = model(nxt, caches)       # ← 只喂【一个新 token】
        tokens_processed += 1
    return ids, tokens_processed


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyLM().eval()

    PROMPT = torch.tensor([[1, 2, 3, 4]])         # 4 个 token 的"提示词"
    N_NEW = 12

    print("=" * 66)
    print("对比两种生成方式（同一个模型、同一段 prompt）")
    print("=" * 66)
    print(f"  prompt 长度 = {PROMPT.shape[1]}，再生成 {N_NEW} 个 token\n")

    ids_a, tok_a = generate_naive(model, PROMPT, N_NEW)
    ids_b, tok_b = generate_cached(model, PROMPT, N_NEW)

    print(f"  A) 每步重算前缀:   生成 {ids_a.shape[1] - PROMPT.shape[1]} 个 token")
    print(f"     ids = {ids_a[0].tolist()}")
    print(f"  B) prefill + cache: 生成 {ids_b.shape[1] - PROMPT.shape[1]} 个 token")
    print(f"     ids = {ids_b[0].tolist()}")
    print(f"\n  两者输出 {'✅ 完全相同' if torch.equal(ids_a, ids_b) else '❌ 不同'}")

    print("\n" + "=" * 66)
    print("计算量：一共让 transformer 处理了多少个 token")
    print("=" * 66)
    print(f"  A) 每步重算前缀:   {tok_a} 个 token   （≈ prompt·n + n²/2，O(n²)）")
    print(f"  B) prefill+cache:  {tok_b} 个 token   （= prompt + n，O(n)）")
    print(f"  → 相差 {tok_a / tok_b:.2f} 倍")

    # 更长的 prompt / 更多 token，差距会更大
    print("\n  放大到更长的 prompt（差距按平方增长）：")
    for plen, n_new in [(4, 12), (32, 32), (128, 128), (512, 512)]:
        p = torch.randint(1, VOCAB_SIZE, (1, plen))
        _, ta = generate_naive(model, p, n_new)
        _, tb = generate_cached(model, p, n_new)
        print(f"    prompt={plen:4d}, 生成={n_new:4d}   A={ta:>7d}  B={tb:>7d}  →  {ta/tb:5.1f} 倍")

    print("\n" + "=" * 66)
    print("映射回 Alpamayo")
    print("=" * 66)
    print("  stage13 的 generate():  每步重喂整条前缀  ← 对应上面的 A")
    print("  真实 alpamayo1_5.py:    VLM 生成时留下 KV cache，Expert 直接复用")
    print("     prompt_cache = vlm_outputs.past_key_values")
    print("     expert(..., past_key_values=prompt_cache)")
    print("  → real2 里那个 offset=3103、序列长 3086，prefix 几千个 token，")
    print("    用 A 的话每步都要重算几千个 —— 这就是 KV cache 存在的意义。")
