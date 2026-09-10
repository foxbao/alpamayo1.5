"""Stage 15: CFG —— 用分类器自由引导放大「左转」指令"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN, VLM_SEQ_LEN,
)

KAPPA = 0.1   # "左转"的目标曲率


class ConditionGenerator(nn.Module):
    """两个可学习的条件：0=左转，1=空条件。"""
    def __init__(self, num_conditions, hidden=HIDDEN, seq_len=VLM_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(num_conditions, hidden)
        self.seq_len = seq_len

    def forward(self, idx):
        return self.embed(idx)[:, None, :].expand(-1, self.seq_len, -1)  # (B,L,H)


class MiniVLA(nn.Module):
    def __init__(self, num_conditions):
        super().__init__()
        self.cond_gen = ConditionGenerator(num_conditions)
        self.in_proj = ActionInProj()
        self.expert = Expert()
        self.out_proj = ActionOutProj()

    def step_fn(self, x, t, condition):
        emb = self.in_proj(x, t)
        h = self.expert(emb, condition)
        return self.out_proj(h)

    def sample_cfg(self, condition, null_condition, w, batch_size, n_steps=10):
        x = torch.randn(batch_size, N_WAYPOINTS, ACTION_DIM)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps)
            v_cond = self.step_fn(x, t, condition)          # 有条件
            v_uncond = self.step_fn(x, t, null_condition)   # 无条件
            v_guided = (1 - w) * v_uncond + w * v_cond      # ← CFG 公式
            x = x + dt * v_guided
        return x   # 返回动作（看 mean κ）


def train(model, opt, n_iters=5000, batch=64):
    left = torch.zeros(N_WAYPOINTS, ACTION_DIM); left[:, 1] = KAPPA   # 左转
    null = torch.zeros(N_WAYPOINTS, ACTION_DIM)                        # 空条件（直行）
    targets = torch.stack([left, null])                                # (2,N,2)

    for it in range(n_iters):
        mode = torch.randint(0, 2, (batch,))            # 50% 左转 / 50% 空条件
        condition = model.cond_gen(mode)
        x1 = targets[mode]
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(batch)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_target = x1 - x0
        v_pred = model.step_fn(x_t, t, condition)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"iter {it:4d}  loss={loss.item():.4f}")


if __name__ == "__main__":
    model = MiniVLA(2)   # 2 个条件
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt)

    left_cond = model.cond_gen(torch.tensor([0]))
    null_cond = model.cond_gen(torch.tensor([1]))

    print("\n不同 guidance weight w 下，采样动作的平均曲率 κ：")
    with torch.no_grad():
        for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
            action = model.sample_cfg(left_cond, null_cond, w, batch_size=1)
            kappa = action[0, :, 1].mean().item()
            print(f"  w={w:.1f}  mean κ = {kappa:+.3f}")
