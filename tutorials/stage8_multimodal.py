"""Stage 8: 多峰分布 —— 同一个条件，多个合理未来（并列分支）

【目的】看清「一个条件 → 多个正确答案」时会发生什么：
  - 场景：路口，左转和右转都合理。目标动作就是 ±KAPPA 两簇
  - **用 MSE 回归单个输出会拟合到条件均值**（κ≈0，即「直行」）——
    而直行恰恰是两簇之间【最不可能】的答案。这就是扩散/生成式建模存在的理由
  - 扩散模型怎么绕开：加噪后同一个条件对应不同的噪声样本，
    采样时从不同噪声出发，自然落进不同的模式（多峰性来自初始噪声，不来自网络输出）
  - 观察方式：采样 20 次，看平均曲率 κ 的两簇分布（注意这只是粗略分组）
  - 对比实验：把 KAPPA 从 0.05 放大到 0.2，让两模式分得更开、更容易观察

【简化】
  - 条件是一个**固定的可学习参数**(1,4,HIDDEN)——「路口」这个条件与输入无关，
    所有样本共用同一份条件。真实条件是 VLM 对图/历史编码出来的
  - 训练数据只有 ±0.2 两条恒定曲率的直线动作（真实轨迹的曲率随时间变化）
  - 只有 2 个模式、20 个采样、CPU 可跑；没有做模式覆盖率的严格统计
  - 判据只按「平均曲率」分组，**没有逐时刻检查动作是否真接近目标轨迹**
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
)

KAPPA = 0.2   # 两个模式的曲率幅度：+0.2 左转，-0.2 右转


class MiniVLA(nn.Module):
    """一个"路口"模型：未来左转或右转都合理。"""
    def __init__(self):
        super().__init__()
        self.cond_seq = nn.Parameter(torch.randn(1, 4, HIDDEN))  # 固定的"路口"条件
        self.in_proj = ActionInProj()
        self.expert = CrossAttnExpert()
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
    model.train()
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
    torch.manual_seed(0)
    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt)
    model.eval()

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
        print("（这是按平均曲率做的粗略分组，还需检查逐时刻动作是否接近目标。）")
        print("（同一条件的单输出 MSE 动作回归会拟合 κ≈0，不表达本例左右两个模式。）")
