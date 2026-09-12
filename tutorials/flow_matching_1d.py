"""appendix / Flow Matching 最小例子：1D 双峰 {-1, +1}（和 ddpm_1d.py 对照）

【目的】把 toy 用的 flow matching 剥到最小，看清训练目标和采样路径本身：
  - **插值**：x_t = (1-t)·x₀ + t·x₁ —— 直线路径，**没有噪声调度**（区别于 DDPM）
  - **训练目标**：v = x₁ - x₀，就是「从噪声指向数据」的直线速度，损失是朴素 MSE
  - **采样**：从高斯噪声出发做欧拉积分（50 步），每步 x += dt·v
  和 `ddpm_1d.py` 的核心对照：
    | | DDPM | Flow Matching（本脚本 / toy 用的） |
    |---|---|---|
    | 插值路径 | x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε（有调度） | x_t = (1-t)·x₀ + t·x₁（直线，无调度） |
    | 预测对象 | 噪声 ε | 速度 v = x₁ - x₀ |
    | 采样 | 反解 x₀ + 后验噪声，500 步 | 欧拉积分，50 步 |
  ⚠️ **符号约定相反，极易混淆**：FM 里 x₀=噪声、x₁=数据（下标是时间 t）；
  DDPM 里 x₀=数据、x_T=噪声（下标是加噪步数）。
  这个「v = x₁ - x₀」和 stage6 训练循环里的 `v_target = x1 - x0` 是同一行代码。

【简化】
  - 1D 双峰，没有图像、没有 transformer、没有时间步嵌入（只把 t 拼接进 MLP）
  - 步数与质量的权衡取决于模型和求解器，**不能**把 50 vs 500 推广成固定速度比
  - 只说明训练目标和采样路径，不代表完整的生成系统

【前提】纯 CPU。附录内容，不属于 stage 递进主线。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorField(nn.Module):
    """输入 (x_t, t)，预测速度 v。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


def train(net, opt, n_iters=5000, batch=128):
    net.train()
    for it in range(n_iters):
        x1 = torch.randint(0, 2, (batch, 1)).float() * 2 - 1   # 数据 ±1
        x0 = torch.randn(batch, 1)                              # 噪声
        t = torch.rand(batch, 1)                                # 随机时间
        xt = (1 - t) * x0 + t * x1                              # ① 直线插值
        v = x1 - x0                                             # ② 真值速度
        pred = net(xt, t)                                       # ③ 预测速度
        loss = F.mse_loss(pred, v)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"iter {it:4d}  loss={loss.item():.4f}")


@torch.no_grad()
def sample(net, n=1000, n_steps=50):
    x = torch.randn(n, 1)                     # 从噪声出发（t=0）
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n, 1), i / n_steps)
        x = x + dt * net(x, t)                # 欧拉积分一步
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    net = VectorField()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train(net, opt)
    net.eval()

    out = sample(net)
    n_neg = (out < -0.5).sum().item()
    n_pos = (out > 0.5).sum().item()
    n_mid = out.numel() - n_neg - n_pos
    print(f"\n采样 {out.numel()} 个：-1 附近 {n_neg}，+1 附近 {n_pos}，中间 {n_mid}")
    print(f"均值 = {out.mean().item():.4f}")
