"""Stage 10: 补 Language —— 文本指令 → condition"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
)

KAPPA = 0.1

# 迷你词表：把"词"映射成 token id（真实里是 tokenizer 干的，这里手写个小的）
VOCAB = {"turn": 0, "left": 1, "right": 2, "continue": 3, "straight": 4}
VOCAB_SIZE = len(VOCAB)

# 三条指令 → token id 序列
INSTRUCTIONS = torch.tensor([
    [0, 1],   # "turn left"
    [0, 2],   # "turn right"
    [3, 4],   # "continue straight"
])   # (3, 2)


class TextEncoder(nn.Module):
    """文本 token 序列 → condition。

    同样需要位置编码——否则「turn left」和「left turn」对模型完全一样。
    """
    def __init__(self, vocab_size=VOCAB_SIZE, hidden=HIDDEN, num_layers=2, max_len=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)          # token id → 词向量
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))   # ← 可学习位置编码
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        x = x + self.pos_embed[:, : x.shape[1]]      # 加上位置编码（真实里用 RoPE）
        return self.encoder(x)                       # (B, L) -> (B, L, HIDDEN)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = TextEncoder()   # 文本 → condition
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

    def sample(self, token_ids):
        self.condition = self.text_enc(token_ids)   # (B, L, HIDDEN)
        action = self.fm.sample(self.step_fn, batch_size=token_ids.shape[0])
        return self.action_space.action_to_traj(action)


def train(model, opt, target_actions, n_iters=5000, batch=64):
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        tokens = INSTRUCTIONS[mode]               # (B, 2)
        model.condition = model.text_enc(tokens)  # (B, 2, HIDDEN)
        x1 = target_actions[mode]
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
    target[0, :, 1] = KAPPA     # left
    target[1, :, 1] = -KAPPA    # right
    target[2, :, 1] = 0.0       # straight

    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)

    print("\n训练后：给文本指令，模型输出轨迹")
    names = ["turn left", "turn right", "continue straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            tokens = INSTRUCTIONS[i:i + 1]       # (1, 2)
            traj = model.sample(tokens)
            end = traj[0, -1, :2]
            print(f"  {name:18s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
