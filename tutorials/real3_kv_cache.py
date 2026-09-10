"""real3: KV cache 实验 —— 为什么真实代码 prefill 一次就够

§六 标的「结构差异①」：toy 每步重算 cross-attention，真实代码把 VLM 的 KV cache
直接复用给 expert。本实验验证：两者【数学等价】，但计算量差 N_steps 倍。

对应真实代码：models/alpamayo1_5.py:304（prompt_cache）与 :349（past_key_values=prompt_cache）
纯 CPU，秒级。
"""

import time

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
        attn = torch.softmax(q @ k.transpose(-1, -2) / HIDDEN**0.5, dim=-1)
        return self.o(attn @ v)


class CrossAttnKVCache(nn.Module):
    """真实写法：条件 K/V 只算一次（prefill），之后复用。"""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.cache = None
        self.kv_calls = 0

    def prefill(self, cond):
        """VLM 跑一次，把条件的 K/V 缓存下来。"""
        self.cache = (self.k(cond), self.v(cond))
        self.kv_calls += 1
        return self.cache

    def forward(self, x):
        k, v = self.cache           # ← 直接复用，不再投影
        q = self.q(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / HIDDEN**0.5, dim=-1)
        return self.o(attn @ v)


def sec(t):
    print(f"\n{'='*64}\n{t}\n{'='*64}")


def run_denoise(module, x, cond, use_cache: bool):
    """跑 N_STEPS 步去噪（简化版：每步只过一次 cross-attn）。"""
    if use_cache:
        module.prefill(cond)          # 只算一次条件的 K/V
    for _ in range(N_STEPS):
        out = module(x) if use_cache else module(x, cond)
        x = x + 0.1 * out
    return x


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
    print(f"  真实写法（prefill）:  {b.kv_calls} 次   ← 只有 1 次")
    print(f"  → 相差 {a.kv_calls / b.kv_calls:.0f} 倍（等于 N_STEPS={N_STEPS}）")

    sec("③ 条件前缀越长，复用的收益越大")
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

    sec("④ 实测耗时（prefix=3072，模拟真实规模）")
    L = 3072
    ca = CrossAttnRecompute()
    cb = CrossAttnKVCache()
    cb.load_state_dict(ca.state_dict())
    c = torch.randn(1, L, HIDDEN)
    xx = torch.randn(1, 64, HIDDEN)

    t0 = time.time(); run_denoise(ca, xx.clone(), c, use_cache=False); t_a = time.time() - t0
    t0 = time.time(); run_denoise(cb, xx.clone(), c, use_cache=True);  t_b = time.time() - t0
    print(f"  toy 写法（每步重算）: {t_a*1000:8.1f} ms")
    print(f"  真实写法（prefill）:  {t_b*1000:8.1f} ms")
    print(f"  → 快 {t_a/t_b:.1f} 倍")
    print("\n  ⚠️ 注意：上面只省了「条件的 K/V 投影」，收益有限（attention 本身还是要算）。")
    print("     真实收益远不止于此 —— 见 ⑤。")

    sec("⑤ 真实收益：缓存省掉的是「整个前缀的前向」")
    print("  真实 expert 是多层 transformer。不用缓存时，每步都要把")
    print("  [前缀 + 动作] 整条序列过一遍；用缓存时，每步只过动作 token。")
    L2, A = 3072, 64
    stack = nn.Sequential(*[
        nn.TransformerEncoderLayer(HIDDEN, 4, HIDDEN * 2, dropout=0.0, batch_first=True)
        for _ in range(4)
    ])
    full = torch.randn(1, L2 + A, HIDDEN)
    act = torch.randn(1, A, HIDDEN)

    t0 = time.time()
    for _ in range(N_STEPS):
        stack(full)
    t_full = time.time() - t0

    t0 = time.time()
    for _ in range(N_STEPS):
        stack(act)
    t_act = time.time() - t0

    print(f"\n  每步过 [前缀+动作] ({L2 + A} tokens): {t_full*1000:8.1f} ms")
    print(f"  每步只过 [动作]    ({A} tokens):      {t_act*1000:8.1f} ms")
    print(f"  → 快 {t_full / t_act:.0f} 倍")
    print(f"\n  理论倍数 ≈ (L+A)/A = {(L2 + A)/A:.0f}（这里 4 层小 transformer 的实测）")
    print("  真实模型 36 层、前缀 ~3100，这个差距会被进一步放大。")
