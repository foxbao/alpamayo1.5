"""Stage 3: 两个投影 + Fourier 编码 —— 动作与 transformer 之间的「桥」

【目的】理解连续动作 (B,64,2) 怎么进出 transformer：
  - ActionInProj：噪声动作 x + 时间步 t → (B,64,HIDDEN)，**每个 waypoint 一个 token**
  - ActionOutProj：(B,64,HIDDEN) → 向量场 (B,64,2)，回到动作空间
  - Fourier 编码：标量 → [sin(2πf_i·x), cos(2πf_i·x)]，f 对数间隔取 1→100
    为什么需要：不加编码时 MLP 很难表示高频/周期函数（同 NeRF 的位置编码）
  - **时间步 t 也必须编码**：否则网络分不清「去噪早期」和「去噪晚期」，
    而不同 t 该输出的向量场完全不同（t≈0 时几乎是纯噪声，t≈1 时接近数据）
  - 结尾打印 t=0.0 与 t=0.9 编码的余弦相似度，直观确认 t 被区分开了

【简化】相对 PerWaypointActionInProjV2：
  - toy 用 3 层 MLP、hidden=64；真实默认 num_enc_layers=4、hidden_size=1024，
    out_dim 对齐 expert hidden（release 为 2048），实际值由 action_in_proj_cfg 覆盖
  - Fourier 特征数 16（=8 个频率）；真实默认 num_fourier_feats=20
  - 真实 MLP 内部用 **RMSNorm**（不是 LayerNorm），归一化位置也略有不同
  - 这里的 OutProj 是 hidden→hidden→2；真实的 action_out_proj 由配置决定
  - 两个投影都是【随机初始化且不训练】的（stage6 起才进入训练循环）
"""

import math
import torch
import torch.nn as nn

BATCH = 2
N_WAYPOINTS = 64
ACTION_DIM = 2
HIDDEN = 64
NUM_FOURIER = 16   # 每个标量的 Fourier 特征数（= 8 个频率的 sin+cos）

class FourierEncoder(nn.Module):
    """把一个标量编码成 Fourier 特征（多频率 sin/cos）。"""
    def __init__(self, num_feats=NUM_FOURIER, max_freq=100.0):
        super().__init__()
        half=num_feats // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half) # 1 → 100 对数间隔
        self.register_buffer("freqs", freqs)   # (half,)
        self.out_dim=num_feats
        
    def forward(self, x):
        arg = x[..., None] * self.freqs * 2 * math.pi   # (..., half)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (..., num_feats)
    
class ActionInProj(nn.Module):
    """噪声动作 (B,64,2) + 时间 t → embedding (B,64,HIDDEN)。"""
    def __init__(self, hidden=HIDDEN, num_fourier=NUM_FOURIER):
        super().__init__()
        self.accel_enc = FourierEncoder(num_fourier)
        self.kappa_enc = FourierEncoder(num_fourier)
        self.time_enc = FourierEncoder(num_fourier)
        in_feats = num_fourier * 3          # accel + kappa + time
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, t):
        B, T, _ = x.shape
        f_accel = self.accel_enc(x[..., 0])                 # (B,T,F)
        f_kappa = self.kappa_enc(x[..., 1])                 # (B,T,F)
        f_t = self.time_enc(t)[:, None, :].expand(B, T, -1) # (B,T,F)
        feat = torch.cat([f_accel, f_kappa, f_t], dim=-1)   # (B,T,3F)
        return self.norm(self.mlp(feat))                    # (B,T,HIDDEN)


        
class ActionOutProj(nn.Module):
    """embedding (B,64,HIDDEN) → 向量场 (B,64,2)。"""
    def __init__(self, hidden=HIDDEN, out_dim=ACTION_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        return self.mlp(h)   # (B,T,HIDDEN) -> (B,T,2)
    
    



if __name__ == "__main__":
    in_proj = ActionInProj()
    out_proj = ActionOutProj()

    x = torch.randn(BATCH, N_WAYPOINTS, ACTION_DIM)
    t = torch.full((BATCH,), 0.3)

    emb = in_proj(x, t)
    v = out_proj(emb)
    print(f"噪声动作 x:   {tuple(x.shape)}")
    print(f"→ embedding:  {tuple(emb.shape)}   ← 每个 waypoint 从 2 维变成 {HIDDEN} 维")
    print(f"→ 向量场 v:   {tuple(v.shape)}")
    
    # 验证时间编码：不同 t 编码出的特征应该很不一样
    enc=FourierEncoder()
    feat_t0=enc(torch.tensor([0.0]))
    feat_t1=enc(torch.tensor([0.9]))
    
    sim=torch.nn.functional.cosine_similarity(feat_t0, feat_t1, dim=-1)
    print(f"\n时间编码 t=0.0 与 t=0.9 的余弦相似度: {sim.item():.3f}")
    print("（接近 0 → 网络能清楚区分'去噪早期'和'去噪晚期'）")


    
