"""Stage 10: 补 Language —— 文本指令 → condition

⚠️ 注意：这里的文本是【输入指令】（外部给定的），不是模型生成的推理。
   真实 Alpamayo 的导航指令长这样（见 notebooks/nav_demo_samples.json）：
       "Turn left in 11m" / "Turn right in 30m" / "Turn left in 4m"
   本 stage 简化为同样的「动作 + 距离」模式，用词刻意和 stage13 的【生成推理】
   （shift/left/due/to/curve...）区分开。

   真实里两者角色不同：
       导航指令（输入）→ 影响推理怎么写；CoC 推理（输出）→ 隐状态影响轨迹怎么出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
)

KAPPA = 0.1

# 迷你词表：把"词"映射成 token id（真实里是 tokenizer 干的，这里手写个小的）
# 模仿真实导航指令 "Turn left in 11m" 的「动作 + 距离」格式
VOCAB = {"turn": 0, "left": 1, "right": 2, "keep": 3, "straight": 4,
         "in": 5, "10m": 6, "30m": 7}
VOCAB_SIZE = len(VOCAB)

# 三条指令 → token id 序列
INSTRUCTIONS = torch.tensor([
    [0, 1, 5, 6],   # "turn left in 10m"      （真实例："Turn left in 11m"）
    [0, 2, 5, 7],   # "turn right in 30m"     （真实例："Turn right in 30m"）
    [3, 4, 5, 7],   # "keep straight in 30m"
])   # (3, 4)


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
    model.train()
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
    torch.manual_seed(0)
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = KAPPA     # left
    target[1, :, 1] = -KAPPA    # right
    target[2, :, 1] = 0.0       # straight

    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)
    model.eval()

    print("\n训练后：给文本指令，模型输出轨迹")
    names = ["turn left in 10m", "turn right in 30m", "keep straight in 30m"]
    with torch.no_grad():
        for i, name in enumerate(names):
            tokens = INSTRUCTIONS[i:i + 1]       # (1, 2)
            traj = model.sample(tokens)
            end = traj[0, -1, :2]
            print(f"  {name:18s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
