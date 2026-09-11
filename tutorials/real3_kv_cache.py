"""real3: KV cache 实验 —— 为什么真实代码 prefill 一次就够

本实验比较同一个 cross-attention 缓存/重算条件 K/V 的结果与投影次数。
它不证明 toy 的 self-attention + cross-attention 与真实 prefix attention 数学等价。

对应真实代码：models/alpamayo1_5.py:304（prompt_cache）与 :349（past_key_values=prompt_cache）
纯 CPU，秒级。
"""

import time
from statistics import median

import torch
import torch.nn as nn

HIDDEN = 128
L_PREFIX = 8        # 条件前缀长度（真实里是几千）
N_STEPS = 10        # 扩散去噪步数


class CrossAttnRecompute(nn.Module):
    """toy 写法：每步都对条件重算 K/V。"""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.kv_calls = 0          # 统计条件的投影算了几次

    def forward(self, x, cond):
        q = self.q(x)
        k = self.k(cond)           # ← 每步重算
        v = self.v(cond)           # ← 每步重算
        self.kv_calls += 1
        attn = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1]**0.5, dim=-1)
        return self.o(attn @ v)


class CrossAttnKVCache(nn.Module):
    """缓存版 cross-attention：条件 K/V 只算一次，之后复用。"""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.cache = None
        self.kv_calls = 0

    def prefill(self, cond):
        """缓存本层的条件 K/V；这里没有实现真实 VLM 的多层 prefill。"""
        self.cache = (self.k(cond), self.v(cond))
        self.kv_calls += 1
        return self.cache

    def forward(self, x):
        k, v = self.cache           # ← 直接复用，不再投影
        q = self.q(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1]**0.5, dim=-1)
        return self.o(attn @ v)


def sec(t):
    print(f"\n{'='*64}\n{t}\n{'='*64}")


@torch.no_grad()
def run_denoise(module, x, cond, use_cache: bool):
    """跑 N_STEPS 步去噪（简化版：每步只过一次 cross-attn）。"""
    if use_cache:
        module.prefill(cond)          # 只算一次条件的 K/V
    for _ in range(N_STEPS):
        out = module(x) if use_cache else module(x, cond)
        x = x + 0.1 * out
    return x


@torch.no_grad()
def benchmark(fn, repeats=3):
    """Warm up, then report median CPU wall time in seconds."""
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return median(times)


if __name__ == "__main__":
    torch.manual_seed(0)

    a = CrossAttnRecompute()
    b = CrossAttnKVCache()
    b.load_state_dict(a.state_dict())      # 共享权重，保证可比

    cond = torch.randn(1, L_PREFIX, HIDDEN)
    x0 = torch.randn(1, 64, HIDDEN)

    sec("① 数值等价性：两种写法输出一样吗？")
    out_a = run_denoise(a, x0.clone(), cond, use_cache=False)
    out_b = run_denoise(b, x0.clone(), cond, use_cache=True)
    diff = (out_a - out_b).abs().max().item()
    print(f"  最大差异 = {diff:.2e}")
    print(f"  torch.allclose → {torch.allclose(out_a, out_b, atol=1e-5)}")
    print("  ✅ 数学等价：复用缓存 = 每步重算（浮点误差以内）")

    sec("② 计算量：条件的 K/V 投影算了几次？")
    print(f"  toy 写法（每步重算）: {a.kv_calls} 次   ← = 扩散步数")
    print(f"  缓存条件 K/V:       {b.kv_calls} 次   ← 只有 1 次")
    print(f"  → 相差 {a.kv_calls / b.kv_calls:.0f} 倍（等于 N_STEPS={N_STEPS}）")

    sec("③ 条件前缀越长，省去的投影 token 数越多")
    print(f"  {'prefix 长度':>12}  {'toy 重算':>12}  {'KV cache':>12}  {'投影 token 数之比':>18}")
    for L in (8, 64, 512, 3072):        # 3072 ≈ 真实序列长度
        ca = CrossAttnRecompute()
        cb = CrossAttnKVCache()
        cb.load_state_dict(ca.state_dict())
        c = torch.randn(1, L, HIDDEN)
        xx = torch.randn(1, 64, HIDDEN)
        run_denoise(ca, xx.clone(), c, use_cache=False)
        run_denoise(cb, xx.clone(), c, use_cache=True)
        tok_a = L * ca.kv_calls          # 投影过的 token 数
        tok_b = L * cb.kv_calls
        print(f"  {L:>12}  {ca.kv_calls:>10} 次  {cb.kv_calls:>10} 次  {tok_a:>8} : {tok_b:<6}")

    sec("④ CPU 示例耗时（prefix=3072，hidden=128；预热后取三次中位数）")
    L = 3072
    ca = CrossAttnRecompute()
    cb = CrossAttnKVCache()
    cb.load_state_dict(ca.state_dict())
    c = torch.randn(1, L, HIDDEN)
    xx = torch.randn(1, 64, HIDDEN)

    t_a = benchmark(lambda: run_denoise(ca, xx.clone(), c, use_cache=False))
    t_b = benchmark(lambda: run_denoise(cb, xx.clone(), c, use_cache=True))
    print(f"  toy 写法（每步重算）: {t_a*1000:8.1f} ms")
    print(f"  缓存条件 K/V:       {t_b*1000:8.1f} ms")
    print(f"  → 快 {t_a/t_b:.1f} 倍")
    print("\n  ⚠️ 注意：上面只省了「条件的 K/V 投影」，收益有限（attention 本身还是要算）。")
    print("     长前缀的序列长度影响见 ⑤ 的示意。")

    sec("⑤ 序列长度对计算量的影响（不同工作负载的示意）")
    print("  下面只比较不同序列长度的 transformer 前向耗时，")
    print("  用来直观展示 prefix 很长时的计算代价；它不是完整的 KV-cache 实现。")
    L2, A = 3072, 64
    stack = nn.Sequential(*[
        nn.TransformerEncoderLayer(HIDDEN, 4, HIDDEN * 2, dropout=0.0, batch_first=True)
        for _ in range(4)
    ])
    full = torch.randn(1, L2 + A, HIDDEN)
    act = torch.randn(1, A, HIDDEN)

    stack.eval()
    with torch.no_grad():
        stack(full)
        stack(act)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            stack(full)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            stack(act)
        t_act = time.perf_counter() - t0

    print(f"\n  每步过 [前缀+动作] ({L2 + A} tokens): {t_full*1000:8.1f} ms")
    print(f"  每步只过 [动作]    ({A} tokens):      {t_act*1000:8.1f} ms")
    print(f"  → 两种工作负载耗时之比 {t_full / t_act:.0f}")
    print(f"\n  token 数比例 = (L+A)/A = {(L2 + A)/A:.0f}")
    print("  只跑动作的分支完全没有读前缀 K/V，漏掉了真实缓存采样的 attention 成本。")
    print("  实际加速还取决于 attention kernel、缓存布局和硬件；不要把这个比例当作端到端 benchmark。")
