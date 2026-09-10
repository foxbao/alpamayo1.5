import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 stage6 的零件（ActionSpace/FlowMatching/投影/Expert 全都不变）
from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN, N_STEPS, DT,
)

HIST_LEN = 16      # 历史步数
HIST_FEATS = 2     # 每步特征 = (dx, dy) 位移增量
KAPPAS = [0.05, -0.05, 0.0]   # 三种"模式"的曲率：左 / 右 / 直


def make_history(mode, hist_len=HIST_LEN, v0=5.0, dt=DT):
    """根据曲率 mode 生成一段历史轨迹。mode: (B,) 张量 -> (B, H, 2)。"""
    kappa = torch.tensor(KAPPAS)[mode]          # (B,)
    theta = torch.zeros_like(kappa)             # (B,)
    deltas = []
    for _ in range(hist_len):
        theta = theta + kappa * v0 * dt
        deltas.append(torch.stack([v0 * torch.cos(theta) * dt,
                                    v0 * torch.sin(theta) * dt], dim=-1))
    return torch.stack(deltas, dim=1)           # (B, H, 2)


class ConditionEncoder(nn.Module):
    """把历史轨迹序列编码成 condition（替代离散 Embedding）。

    注意：nn.TransformerEncoder 本身**不含**位置编码，必须自己加——
    否则序列顺序对模型没有意义（「先左后右」和「先右后左」会长得一样）。
    """
    def __init__(self, hist_feats=HIST_FEATS, hidden=HIDDEN, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.proj = nn.Linear(hist_feats, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))   # ← 可学习位置编码
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, history):
        # history: (B, H, 2) -> condition: (B, H, HIDDEN)
        x = self.proj(history)
        x = x + self.pos_embed[:, : x.shape[1]]      # 加上位置编码（真实里用 RoPE）
        return self.encoder(x)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_enc = ConditionEncoder()   # ← 关键变化：Encoder 替代 Embedding
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

    def sample(self, history):
        self.condition = self.cond_enc(history)
        action = self.fm.sample(self.step_fn, batch_size=history.shape[0], temperature=0.3)
        return self.action_space.action_to_traj(action)



def train(model, opt, target_actions, n_iters=8000, batch=32):
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))               # 随机模式
        hist = make_history(mode)                          # (B,H,2) 生成历史轨迹
        model.condition = model.cond_enc(hist)             # (B,H,H) 编码成 condition
        x1 = target_actions[mode]                          # 目标动作
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
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = 0.05
    target[1, :, 1] = -0.05
    target[2, :, 1] = 0.0

    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)

    print("\n训练后：给一段历史轨迹，模型自己推断未来方向")
    with torch.no_grad():
        for name, m in [("left", 0), ("right", 1), ("straight", 2)]:
            hist = make_history(torch.tensor([m]))     # (1, H, 2)
            traj = model.sample(hist)
            end = traj[0, -1, :2]
            print(f"  {name:8s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
