"""Flow Matching 最小例子：1D 双峰 {-1,+1}（和 stage9 DDPM 对照）"""

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
    net = VectorField()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train(net, opt)

    out = sample(net)
    n_neg = (out < -0.5).sum().item()
    n_pos = (out > 0.5).sum().item()
    n_mid = out.numel() - n_neg - n_pos
    print(f"\n采样 {out.numel()} 个：-1 附近 {n_neg}，+1 附近 {n_pos}，中间 {n_mid}")
    print(f"均值 = {out.mean().item():.4f}")
