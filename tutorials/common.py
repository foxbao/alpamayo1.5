"""共享积木：stage4 之后各 stage 复用的模块。

包含：常量 + ActionSpace + FlowMatching + 投影（Fourier / InProj / OutProj）+ Expert。
用法：`from common import ActionSpace, FlowMatching, ...`

对应关系：
- ActionSpace        ← stage1（动作空间）
- FlowMatching       ← stage2（扩散采样）
- FourierEncoder 等  ← stage3（投影）
- Expert / ExpertBlock ← stage4（cross-attention 去噪器）
"""

import math

import torch
import torch.nn as nn

N_WAYPOINTS = 64
ACTION_DIM = 2
HIDDEN = 64
NUM_FOURIER = 16
N_STEPS = 10
VLM_SEQ_LEN = 32
DT = 0.1


# ---- ActionSpace（stage1）：动作 -> 轨迹 ----
class ActionSpace:
    def action_to_traj(self, action, v0=5.0):
        # toy 里 mean=0/std=1，反归一化是恒等（×1+0），故省略；真实代码有非平凡归一化
        accel = action[..., 0]
        kappa = action[..., 1].clamp(-0.33, 0.33)
        v = torch.full_like(accel[:, :1], v0)
        theta = torch.zeros_like(accel[:, :1])
        x = torch.zeros_like(accel[:, :1]); y = torch.zeros_like(accel[:, :1])
        xs, ys = [], []
        for t in range(N_WAYPOINTS):
            v = v + accel[:, t:t + 1] * DT
            theta = theta + kappa[:, t:t + 1] * v * DT
            x = x + v * torch.cos(theta) * DT
            y = y + v * torch.sin(theta) * DT
            xs.append(x); ys.append(y)
        x = torch.cat(xs, -1); y = torch.cat(ys, -1)
        return torch.stack([x, y, torch.zeros_like(x)], -1)


# ---- FlowMatching（stage2）：扩散采样，支持任意 batch ----
class FlowMatching:
    def __init__(self, n_steps=N_STEPS):
        self.n_steps = n_steps

    def sample(self, step_fn, batch_size, temperature=1.0):
        x = torch.randn(batch_size, N_WAYPOINTS, ACTION_DIM) * temperature
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = torch.full((batch_size,), i / self.n_steps)
            x = x + dt * step_fn(x, t)
        return x


# ---- 投影（stage3）：动作 <-> embedding，Fourier 编码 ----
class FourierEncoder(nn.Module):
    def __init__(self, num_feats=NUM_FOURIER, max_freq=100.0):
        super().__init__()
        half = num_feats // 2
        self.register_buffer("freqs", torch.logspace(0, math.log10(max_freq), steps=half))
        self.out_dim = num_feats

    def forward(self, x):
        arg = x[..., None] * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1)


class ActionInProj(nn.Module):
    def __init__(self, hidden=HIDDEN, num_fourier=NUM_FOURIER):
        super().__init__()
        self.accel_enc = FourierEncoder(num_fourier)
        self.kappa_enc = FourierEncoder(num_fourier)
        self.time_enc = FourierEncoder(num_fourier)
        self.mlp = nn.Sequential(nn.Linear(num_fourier * 3, hidden), nn.SiLU(),
                                nn.Linear(hidden, hidden), nn.SiLU(),
                                nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, t):
        B, T, _ = x.shape
        f_a = self.accel_enc(x[..., 0]); f_k = self.kappa_enc(x[..., 1])
        f_t = self.time_enc(t)[:, None, :].expand(B, T, -1)
        return self.norm(self.mlp(torch.cat([f_a, f_k, f_t], -1)))


class ActionOutProj(nn.Module):
    def __init__(self, hidden=HIDDEN, out_dim=ACTION_DIM):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                nn.Linear(hidden, out_dim))

    def forward(self, h):
        return self.mlp(h)


# ---- Expert（stage4）：cross-attention 去噪器 ----
class ExpertBlock(nn.Module):
    def __init__(self, hidden=HIDDEN, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(hidden); self.n2 = nn.LayerNorm(hidden); self.n3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden))

    def forward(self, x, cond):
        x = self.n1(x + self.self_attn(x, x, x)[0])
        x = self.n2(x + self.cross_attn(x, cond, cond)[0])
        x = self.n3(x + self.ffn(x))
        return x


class Expert(nn.Module):
    def __init__(self, hidden=HIDDEN, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([ExpertBlock(hidden) for _ in range(n_blocks)])

    def forward(self, x, cond):
        for b in self.blocks:
            x = b(x, cond)
        return x
