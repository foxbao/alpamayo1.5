"""exp: Fourier 编码到底有什么用？—— 用实验说话（读 stage3 之后做）

【目的】用**对照实验**回答「为什么要编码，直接喂标量不行吗」：
  同一个任务、同一个网络结构、同样的训练步数，只改「输入怎么给网络」：
    A) 直接喂标量 x
    B) 先做 Fourier 编码再喂
  看谁能学会一个【高频】函数 f(x) = sin(2π·12·x)（12 个周期）。
  这就是 stage3 里 `FourierEncoder` 存在的理由 —— 不加编码时 MLP 有
  **谱偏差**（spectral bias），会优先拟合低频，学不动高频细节。
  这正是 NeRF / diffusion 的时间步编码采用同一套做法的原因。

【简化】
  - 1D 标量回归，任务与驾驶无关；只演示「能表示高频」这一件事
  - 目标函数是解析给的，没有真实数据分布
  - 结论只说明编码【有助于】拟合高频，不构成对任意网络/任务的普适保证：
    网络宽度、训练步数、频率范围（MAX_FREQ）都会影响结果，可自行改参数验证

【前提】纯 CPU，几秒钟。不属于 stage 主线，默认不纳入回归测试。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FREQ = 8          # 每个标量展开成 2*N_FREQ 个特征（sin + cos）
MAX_FREQ = 100.0
FUNC_FREQ = 12      # 目标函数 f(x)=sin(2π·12·x)，12 个周期 —— 比较高频


def target_fn(x):
    """要拟合的目标：高频正弦。"""
    return torch.sin(2 * math.pi * FUNC_FREQ * x)


class FourierEncoder(nn.Module):
    """把标量 x 展开成多频率 sin/cos（和 stage3/common.py 里一致）。"""
    def __init__(self, n_freq=N_FREQ, max_freq=MAX_FREQ):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(0, math.log10(max_freq), steps=n_freq))
        self.out_dim = 2 * n_freq

    def forward(self, x):                       # (N,) -> (N, 2*n_freq)
        arg = x[:, None] * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class MLP(nn.Module):
    """两种版本共用的网络结构。"""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval(model, x, y, n_iters=3000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(n_iters):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(x)
        mse = F.mse_loss(pred, y).item()
        # 用相关系数看「形状学得像不像」（对幅度不敏感）
        corr = torch.corrcoef(torch.stack([pred, y]))[0, 1].item()
    return mse, corr


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 400)
    y = target_fn(x)

    print(f"目标函数: sin(2π·{FUNC_FREQ}·x)   —— {FUNC_FREQ} 个完整周期")
    print(f"训练 3000 步，网络结构完全相同（2 层 64 单元）\n")

    # ---- A: 直接喂标量 ----
    a = MLP(in_dim=1)
    mse_a, corr_a = train_eval(a, x[:, None], y)

    # ---- B: Fourier 编码后喂 ----
    enc = FourierEncoder()
    xf = enc(x)
    b = MLP(in_dim=xf.shape[-1])
    mse_b, corr_b = train_eval(b, xf, y)

    print(f"{'方案':<28} {'MSE':>10} {'相关系数':>10}")
    print("-" * 52)
    print(f"{'A) 直接喂标量 x':<28} {mse_a:>10.5f} {corr_a:>10.3f}")
    print(f"{'B) Fourier 编码后':<28} {mse_b:>10.5f} {corr_b:>10.3f}")
    print()
    print("相关系数越接近 1 = 形状学得越像")
    print()
    print("看看 A 学出来的形状（每 10 个点取一个）：")
    with torch.no_grad():
        pa = a(x[:, None])
    for i in range(0, 60, 12):
        print(f"  x={x[i]:.3f}  真值={y[i]:+.3f}  A(直接)={pa[i]:+.3f}  B(傅里叶)={b(xf)[i]:+.3f}")
