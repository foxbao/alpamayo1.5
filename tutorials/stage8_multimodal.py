import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
)

KAPPA = 0.2   # 两个模式的曲率幅度：+0.2 左转，-0.2 右转


class MiniVLA(nn.Module):
    """一个"路口"模型：未来左转或右转都合理。"""
    def __init__(self):
        super().__init__()
        self.cond_seq = nn.Parameter(torch.randn(1, 4, HIDDEN))  # 固定的"路口"条件
        self.in_proj = ActionInProj()
        self.expert = Expert()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()
        self.condition = None

    def step_fn(self, x, t):
        emb = self.in_proj(x, t)
        h = self.expert(emb, self.condition)
        return self.out_proj(h)

    def sample_action(self, batch_size):
        self.condition = self.cond_seq.expand(batch_size, -1, -1)
        return self.fm.sample(self.step_fn, batch_size=batch_size)


def train(model, opt, n_iters=5000, batch=64):
    left = torch.zeros(N_WAYPOINTS, ACTION_DIM); left[:, 1] = KAPPA
    right = torch.zeros(N_WAYPOINTS, ACTION_DIM); right[:, 1] = -KAPPA
    targets = torch.stack([left, right], 0)   # (2, N, 2)

    for it in range(n_iters):
        mode = torch.randint(0, 2, (batch,))                 # 随机左/右
        model.condition = model.cond_seq.expand(batch, -1, -1)
        x1 = targets[mode]
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(batch)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_target = x1 - x0
        v_pred = model.step_fn(x_t, t)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"iter {it:4d}  loss={loss.item():.4f}")


if __name__ == "__main__":
    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt)

    print("\n采样 20 次，看动作曲率 κ 的分布：")
    with torch.no_grad():
        action = model.sample_action(20)
        kappas = action[:, :, 1].mean(dim=1).numpy()   # 每条的平均曲率
        for i, k in enumerate(kappas):
            print(f"  样本{i:2d}: κ = {k:+.3f}")

        n_left = int((kappas > 0.05).sum())
        n_right = int((kappas < -0.05).sum())
        n_mid = 20 - n_left - n_right
        print(f"\n左转(κ>+0.05): {n_left}  右转(κ<-0.05): {n_right}  中间: {n_mid}")
        print(f"平均 κ = {kappas.mean():+.4f}")
        print("（左转右转都有、中间≈0 = 学会了多模态分布）")
        print("（若只做回归，输出会是平均 κ≈0 的'直行'——没人要的轨迹）")
