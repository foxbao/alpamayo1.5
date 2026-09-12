"""appendix / DDPM 最小例子：1D 双峰分布 {-1, +1}（和 flow_matching_1d.py 对照）

【目的】补齐扩散的**另一条流派**，好和 toy 用的 flow matching 对照着理解。
  DDPM 的三个特征（本脚本每一处都对应代码）：
    - **前向加噪**：x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε，有 500 步的噪声调度（β 线性从 1e-4 到 0.02）
    - **预测对象**：网络学的是【加进去的噪声 ε】，不是速度、也不是 x₀
    - **反向采样**：先用 ε 反解 x₀，再按后验均值/方差采样（**每步注入随机噪声**），T=500 步
  和 `flow_matching_1d.py` 的核心对照：
    | | DDPM | Flow Matching（toy 用的） |
    |---|---|---|
    | 插值路径 | x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε（有调度） | x_t = (1-t)·x₀ + t·x₁（直线，无调度） |
    | 预测对象 | 噪声 ε | 速度 v = x₁ - x₀ |
    | 采样 | 反解 x₀ + 后验噪声，500 步 | 欧拉积分，50 步 |
  ⚠️ **符号约定相反，极易混淆**：FM 里 x₀=噪声、x₁=数据（下标是时间 t）；
  DDPM 里 x₀=数据、x_T=噪声（下标是加噪步数）。

【简化】
  - 1D 双峰，没有图像、没有 U-Net、没有时间步嵌入（只把 t 拼接进 MLP）
  - 步数与质量的权衡取决于模型和求解器，**不能**把 500 vs 50 推广成固定速度比
  - 目标只是看清「训练目标」和「采样路径」，不代表任何完整的图像生成系统

【前提】纯 CPU。附录内容，不属于 stage 递进主线。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

T=500
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)  # ᾱ_t = 累积"还剩多少信号"
alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
posterior_coef_x0 = beta * alpha_bar_prev.sqrt() / (1 - alpha_bar)
posterior_coef_xt = (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar)

class Denoiser(nn.Module):
    """输入 (x_t, t)，预测加进去的噪声 ε。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))   # t 归一化到 [0,1]


def train(net, opt, n_iters=5000, batch=128):
    net.train()
    for it in range(n_iters):
        x0 = torch.randint(0, 2, (batch, 1)).float() * 2 - 1   # 数据：±1
        t = torch.randint(0, T, (batch, 1))
        eps = torch.randn(batch, 1)
        at = alpha_bar[t]
        xt = at.sqrt() * x0 + (1 - at).sqrt() * eps            # ① 前向加噪
        pred = net(xt, t.float() / (T - 1))                    # ② 预测噪声
        loss = F.mse_loss(pred, eps)                           # ③ 和真噪声比
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"iter {it:4d}  loss={loss.item():.4f}")


@torch.no_grad()
def reverse_step(x_t, eps_pred, t, noise=None):
    """Sample one DDPM reverse transition p(x_{t-1} | x_t)."""
    at = alpha_bar[t]
    x0 = (x_t - (1 - at).sqrt() * eps_pred) / at.sqrt().clamp(min=1e-4)
    if t == 0:
        return x0

    if noise is None:
        noise = torch.randn_like(x_t)
    # q(x_{t-1} | x_t, x_0) 的后验均值和方差（真正的 DDPM 反向一步）。
    mean = posterior_coef_x0[t] * x0 + posterior_coef_xt[t] * x_t
    return mean + posterior_var[t].clamp(min=1e-20).sqrt() * noise


@torch.no_grad()
def sample(net, n=1000):
    x = torch.randn(n, 1)                       # 从纯噪声出发
    for t in reversed(range(T)):
        tt = torch.full((n, 1), t / (T - 1))
        eps = net(x, tt)                        # 预测噪声
        x = reverse_step(x, eps, t)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    net = Denoiser()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train(net, opt)
    net.eval()

    out = sample(net)
    n_neg = (out < -0.5).sum().item()
    n_pos = (out > 0.5).sum().item()
    n_mid = out.numel() - n_neg - n_pos
    print(f"\n采样 {out.numel()} 个：-1 附近 {n_neg}，+1 附近 {n_pos}，中间 {n_mid}")
    print(f"均值 = {out.mean().item():.4f}")
