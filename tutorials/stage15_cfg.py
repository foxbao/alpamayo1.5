"""Stage 15: CFG —— 用分类器自由引导放大「左转」指令

教学近似：toy 学了一个「可学习的空条件 embedding」来表示「无指令」。
真实代码是**从输入序列里删掉 <|route_start|>...<|route_end|> 那一段**
（nav_utils.remove_nav_text），再跑一遍 VLM 得到无条件的 KV cache。
机制相同，构造方式不同。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN, VLM_SEQ_LEN,
)

KAPPA = 0.1   # "左转"的目标曲率


class ConditionGenerator(nn.Module):
    """三个可学习的条件：左转、右转和训练时条件被丢弃后的空条件。"""
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
        self.expert = CrossAttnExpert()
        self.out_proj = ActionOutProj()

    def step_fn(self, x, t, condition):
        emb = self.in_proj(x, t)
        h = self.expert(emb, condition)
        return self.out_proj(h)

    def sample_cfg(self, condition, null_condition, w, batch_size, n_steps=10, noise=None):
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if noise is None:
            noise = torch.randn(batch_size, N_WAYPOINTS, ACTION_DIM)
        if noise.shape != (batch_size, N_WAYPOINTS, ACTION_DIM):
            raise ValueError("noise must have shape (batch_size, N_WAYPOINTS, ACTION_DIM)")

        def expand_condition(value):
            if value.shape[0] == 1:
                return value.expand(batch_size, -1, -1)
            if value.shape[0] != batch_size:
                raise ValueError("condition batch dimension must be 1 or batch_size")
            return value

        condition = expand_condition(condition)
        null_condition = expand_condition(null_condition)
        x = noise.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps)
            v_cond = self.step_fn(x, t, condition)          # 有条件
            v_uncond = self.step_fn(x, t, null_condition)   # 无条件
            v_guided = (1 - w) * v_uncond + w * v_cond      # ← CFG 公式
            x = x + dt * v_guided
        return x   # 返回动作（看 mean κ）


def train(model, opt, n_iters=5000, batch=64):
    model.train()
    left = torch.zeros(N_WAYPOINTS, ACTION_DIM); left[:, 1] = KAPPA
    right = torch.zeros(N_WAYPOINTS, ACTION_DIM); right[:, 1] = -KAPPA
    targets = torch.stack([left, right])  # (2, N, 2)

    for it in range(n_iters):
        mode = torch.randint(0, 2, (batch,))             # 目标：左转或右转
        # CFG 的训练方式：保留原目标，但随机删掉条件，让空条件学习边缘分布。
        condition_ids = mode.clone()
        condition_ids[torch.rand(batch) < 0.5] = 2       # 2 = 空条件
        condition = model.cond_gen(condition_ids)
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
    torch.manual_seed(0)
    model = MiniVLA(3)   # 左转、右转、空条件
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt)
    model.eval()

    left_cond = model.cond_gen(torch.tensor([0]))
    null_cond = model.cond_gen(torch.tensor([2]))

    print("\n固定同一份初始噪声，不同 guidance weight w 下的平均曲率 κ：")
    print("w=0 采边缘分布，并非直行；w>1 外推的是向量场，不保证曲率单调增大。")
    initial_noise = torch.randn(1, N_WAYPOINTS, ACTION_DIM)
    with torch.no_grad():
        for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
            action = model.sample_cfg(
                left_cond, null_cond, w, batch_size=1, noise=initial_noise
            )
            kappa = action[0, :, 1].mean().item()
            print(f"  w={w:.1f}  mean κ = {kappa:+.3f}")
