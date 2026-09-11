"""Stage 11: 融合 —— 历史 + 图片 + 文本 → 一个 condition

⚠️ 本 stage 演示的是「融合【怎么写】」（concat 成一条序列），
   **不能证明「融合为什么必要」**。

   本 toy 的三路输入【编码的是同一个方向】（都取自同一个 mode 0/1/2），
   所以任一路单独给都够用，融合在这里是【冗余】的。

   真实的 VLA 里三路是【互补】的：
       图片 → 场景（车道线、障碍、路牌）
       历史 → 运动状态（速度、朝向）
       文本 → 外部意图（导航指令）
   要验证「融合的必要性」，得让三路承载不同信息做对照——可参考 real6 的消融做法。"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
    build_vit, HistoryEncoder,
)

KAPPA = 0.1
IMG_SIZE = 16
HIST_LEN = 8

# ---------- 三个输入的数据（都编码同一个方向，冗余但演示融合）----------
VOCAB = {"turn": 0, "left": 1, "right": 2, "continue": 3, "straight": 4}
VOCAB_SIZE = len(VOCAB)
INSTRUCTIONS = torch.tensor([[0, 1], [0, 2], [3, 4]])   # (3, 2)


def make_image(mode):
    img = torch.zeros(1, IMG_SIZE, IMG_SIZE)
    for i in range(IMG_SIZE):
        if mode == 0:
            col = max(0, 8 - i // 2)
        elif mode == 1:
            col = min(IMG_SIZE - 1, 8 + i // 2)
        else:
            col = 8
        for d in range(3):
            c = min(IMG_SIZE - 1, max(0, col + d - 1))
            img[0, i, c] = 1.0
    return img


IMAGES = torch.stack([make_image(0), make_image(1), make_image(2)])   # (3,1,16,16)


def make_history(mode):
    kappa = [KAPPA, -KAPPA, 0.0][mode]
    theta = 0.0
    ds = []
    for _ in range(HIST_LEN):
        theta += kappa * 5.0 * 0.1
        ds.append([5.0 * math.cos(theta) * 0.1, 5.0 * math.sin(theta) * 0.1])
    return torch.tensor(ds)


HISTORIES = torch.stack([make_history(0), make_history(1), make_history(2)])  # (3,8,2)


# ---------- 三个编码器（HistoryEncoder / build_vit 已在 common.py，这里只需 TextEncoder）----------
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN)
    def forward(self, tokens):
        return self.embed(tokens)   # (B,2) -> (B,2,64)


# ---------- 融合 + MiniVLA ----------
class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_enc = HistoryEncoder()
        self.vis_enc = build_vit()
        self.text_enc = TextEncoder()
        self.pos_embed = nn.Parameter(torch.empty(1, HIST_LEN + 17 + 2, HIDDEN))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.in_proj = ActionInProj()
        self.expert = Expert()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()
        self.condition = None

    def encode(self, hist, img, text):
        h = self.hist_enc(hist)                    # (B, 8,  64)
        v = self.vis_enc(img).last_hidden_state    # (B, 17, 64)
        t = self.text_enc(text)                    # (B, 2,  64)
        # 固定槽位区分模态，并保留历史/文本顺序；裸 concat 不携带位置信息。
        return torch.cat([h, v, t], dim=1) + self.pos_embed  # (B, 27, 64)

    def step_fn(self, x, t):
        emb = self.in_proj(x, t)
        h = self.expert(emb, self.condition)
        return self.out_proj(h)

    def sample(self, hist, img, text):
        self.condition = self.encode(hist, img, text)
        action = self.fm.sample(self.step_fn, batch_size=hist.shape[0])
        return self.action_space.action_to_traj(action)


def train(model, opt, target_actions, n_iters=5000, batch=64):
    model.train()
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        hist = HISTORIES[mode]
        img = IMAGES[mode]
        text = INSTRUCTIONS[mode]
        model.condition = model.encode(hist, img, text)   # (B,27,64)
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
    target[0, :, 1] = KAPPA
    target[1, :, 1] = -KAPPA
    target[2, :, 1] = 0.0

    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)
    model.eval()

    print("\n训练后：三路输入一起给，模型输出轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj = model.sample(HISTORIES[i:i+1], IMAGES[i:i+1], INSTRUCTIONS[i:i+1])
            end = traj[0, -1, :2]
            print(f"  {name:8s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
