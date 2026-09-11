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
from transformers import ViTConfig, ViTModel

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

    def forward(self, x, cond, cond_pad_mask=None):
        # cond_pad_mask: (B, L_cond) 的 bool，True 表示该 key 位置要被忽略
        # （变长文本的 <pad> 不能参与 attention——见下面的 Expert 说明）
        x = self.n1(x + self.self_attn(x, x, x)[0])
        x = self.n2(x + self.cross_attn(x, cond, cond, key_padding_mask=cond_pad_mask)[0])
        x = self.n3(x + self.ffn(x))
        return x


class Expert(nn.Module):
    """去噪专家（**cross-attention 版** —— 注意：这不是 Alpamayo 的做法）。

    ⚠️ 这里的 ExpertBlock 是「self-attn + cross-attn + FFN」，条件作为单独的
       `cond` 张量传入。这是【通用的条件化做法】（Stable Diffusion、原始
       Transformer decoder 都这样），**但 Alpamayo 的 Expert 没有 cross-attention**：
       它和 VLM 文本塔结构完全相同，动作 token「续写」在 VLM 的逐层 K/V 后面。

       两种做法的对比见 stage4_expert.py；真实做法的完整实现见 exp_prefix_expert.py；
       为什么 toy 选了这一版见 README §六「两种设计哲学」。

       下面这段位置编码说明同样适用于真实 Expert。

    注意：动作 token 需要【位置编码】——否则 64 个 waypoint 在 self-attention 里
    只是一个集合，第 5 个点分不清自己在第 10 个点前面。真实代码同样给 expert 传
    `position_ids`（alpamayo1_5.py:162 `_build_expert_pos_ids_and_attn_mask`）。
    条件序列的顺序/身份信息由各输入编码器提供；cross-attention 本身不区分
    K/V token 的排列，仅把未经编码的标签放在图片旁边不能建立对应关系。
    """
    def __init__(self, hidden=HIDDEN, n_blocks=2, max_len=128):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))   # ← 动作序列的位置编码
        self.blocks = nn.ModuleList([ExpertBlock(hidden) for _ in range(n_blocks)])

    def forward(self, x, cond, cond_pad_mask=None):
        """cond_pad_mask: (B, L_cond) bool，True = 忽略该条件位置（如文本的 <pad>）。

        为什么需要它：`cross_entropy(ignore_index=...)` 只让 pad **不参与 loss**，
        但 pad 仍然进了 transformer、产生了 hidden，也仍然被 cross-attention 读到。
        要真正屏蔽，得在 attention 层用 key_padding_mask。
        真实代码里对应 `_build_expert_pos_ids_and_attn_mask`（alpamayo1_5.py:162）。
        """
        if x.shape[1] > self.pos_embed.shape[1]:
            raise ValueError("Input sequence exceeds Expert max_len")
        x = x + self.pos_embed[:, : x.shape[1]]      # 加上位置编码
        for b in self.blocks:
            x = b(x, cond, cond_pad_mask)
        return x


# ---- 视觉编码（stage9）----
def build_vit():
    """Transformers ViT 结构，随机初始化；16×16 灰度图、patch4，不下载预训练权重。"""
    cfg = ViTConfig(image_size=16, patch_size=4, num_channels=1,
                    hidden_size=HIDDEN, num_hidden_layers=4, num_attention_heads=4,
                    intermediate_size=128, num_labels=0)
    return ViTModel(cfg)


# ---- 历史编码（stage7/11/14）----
class HistoryEncoder(nn.Module):
    """历史轨迹 → tokens（简单投影，复杂处理交给 CosmosReason）。"""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.proj = nn.Linear(2, hidden)

    def forward(self, hist):
        return self.proj(hist)   # (B, H, 2) -> (B, H, HIDDEN)


# ---- 因果 transformer（stage13 的 Cosmos-Reason）----
def make_causal_mask(L):
    """因果 mask：位置 i 只能看 0..i（下三角=0，上三角=-inf）。"""
    return torch.triu(torch.full((L, L), float("-inf")), diagonal=1)


class CausalBlock(nn.Module):
    def __init__(self, hidden=HIDDEN, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden))
        self.n2 = nn.LayerNorm(hidden)

    def forward(self, x, mask):
        x = x + self.attn(x, x, x, attn_mask=mask)[0]
        x = self.n1(x)
        x = x + self.ffn(x)
        x = self.n2(x)
        return x


class CosmosReason(nn.Module):
    """迷你 Cosmos-Reason：因果 transformer，自回归生成推理。"""
    def __init__(self, vocab_size, hidden=HIDDEN, n_blocks=3, max_len=128):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Parameter(torch.empty(1, max_len, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_embeds, text_ids, mask):
        text = self.text_embed(text_ids)              # (B, L, H)
        x = torch.cat([input_embeds, text], dim=1)    # (B, N+L, H)
        if x.shape[1] > self.pos_embed.shape[1]:
            raise ValueError("Input sequence exceeds CosmosReason max_len")
        # 全序列共享绝对位置；causal mask 控制可见性，位置编码标识历史/图像/文本的位置。
        x = x + self.pos_embed[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x, mask)
        return x
