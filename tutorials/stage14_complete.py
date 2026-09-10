"""Stage 14: 完整输入 + CoC —— 历史 + 图片一起进 CosmosReason（最接近真实 Alpamayo）

对比 stage13：这里把「历史轨迹」也作为输入，和历史轨迹编码一起拼进 CosmosReason
的输入序列，再自回归生成 CoC。这才是真实 VLM 的完整输入侧。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
)

KAPPA = 0.1
IMG_SIZE = 16
HIST_LEN = 8          # 历史步数
VISUAL_TOKENS = 17    # ViT 输出：1 CLS + 16 patch

# CoC 词表 + 推理链（同 stage13）
VOCAB = {"<bos>": 0, "turn": 1, "left": 2, "right": 3, "continue": 4, "straight": 5, "<eos>": 6}
VOCAB_SIZE = len(VOCAB)
BOS_ID = 0
REASONING = torch.tensor([[1, 2, 6], [1, 3, 6], [4, 5, 6]])  # turn left / turn right / continue straight


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


def make_history(mode):
    """历史轨迹：和图片同方向的一段位移增量序列。"""
    kappa = [KAPPA, -KAPPA, 0.0][mode]
    theta = 0.0
    ds = []
    for _ in range(HIST_LEN):
        theta += kappa * 5.0 * 0.1
        ds.append([5.0 * math.cos(theta) * 0.1, 5.0 * math.sin(theta) * 0.1])
    return torch.tensor(ds)   # (HIST_LEN, 2)


IMAGES = torch.stack([make_image(0), make_image(1), make_image(2)])        # (3,1,16,16)
HISTORIES = torch.stack([make_history(0), make_history(1), make_history(2)])  # (3,HIST_LEN,2)


def build_vit():
    cfg = ViTConfig(image_size=16, patch_size=4, num_channels=1,
                    hidden_size=HIDDEN, num_hidden_layers=4, num_attention_heads=4,
                    intermediate_size=128, num_labels=0)
    return ViTModel(cfg)


def make_causal_mask(L):
    return torch.triu(torch.full((L, L), float("-inf")), diagonal=1)


class HistoryEncoder(nn.Module):
    """历史轨迹 → tokens（简单投影，复杂处理交给 CosmosReason）。"""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.proj = nn.Linear(2, hidden)

    def forward(self, hist):
        return self.proj(hist)   # (B, HIST_LEN, 2) -> (B, HIST_LEN, HIDDEN)


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
    """因果 transformer：吃 [历史+视觉] 连续 token + 推理 token，自回归生成 CoC。"""
    def __init__(self, vocab_size=VOCAB_SIZE, hidden=HIDDEN, n_blocks=3):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([CausalBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_embeds, text_ids, mask):
        text = self.text_embed(text_ids)              # (B, L, H)
        x = torch.cat([input_embeds, text], dim=1)    # (B, N+L, H)
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_enc = HistoryEncoder()
        self.vit = build_vit()
        self.cosmos = CosmosReason()
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

    def generate(self, hist, img):
        """历史 + 图片 → 融合 → 自回归生成 CoC → 隐状态 = condition。"""
        B = hist.shape[0]
        hist_embeds = self.hist_enc(hist)                    # (B, HIST_LEN, 64)
        visual = self.vit(img).last_hidden_state             # (B, 17, 64)
        input_embeds = torch.cat([hist_embeds, visual], dim=1)  # (B, HIST_LEN+17, 64) 融合

        text_ids = torch.full((B, 1), BOS_ID, dtype=torch.long)
        for _ in range(3):                                   # 生成 3 个 token
            L = input_embeds.shape[1] + text_ids.shape[1]
            h = self.cosmos(input_embeds, text_ids, make_causal_mask(L))
            next_tok = self.cosmos.head(h[:, -1]).argmax(-1, keepdim=True)
            text_ids = torch.cat([text_ids, next_tok], dim=1)

        L = input_embeds.shape[1] + text_ids.shape[1]
        h = self.cosmos(input_embeds, text_ids, make_causal_mask(L))
        self.condition = h[:, input_embeds.shape[1]:]        # 推理部分的隐状态
        return text_ids

    def sample(self, hist, img):
        text_ids = self.generate(hist, img)
        action = self.fm.sample(self.step_fn, batch_size=hist.shape[0])
        traj = self.action_space.action_to_traj(action)
        return traj, text_ids


def train(model, opt, target_actions, n_iters=5000, batch=64):
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        hist = HISTORIES[mode]
        img = IMAGES[mode]
        reasoning = REASONING[mode]

        # ① CosmosReason：teacher forcing
        hist_embeds = model.hist_enc(hist)                   # (B,H,64)
        visual = model.vit(img).last_hidden_state            # (B,17,64)
        input_embeds = torch.cat([hist_embeds, visual], dim=1)
        bos = torch.full((batch, 1), BOS_ID, dtype=torch.long)
        text_in = torch.cat([bos, reasoning[:, :-1]], dim=1)  # [bos, turn, left]
        L = input_embeds.shape[1] + text_in.shape[1]
        h = model.cosmos(input_embeds, text_in, make_causal_mask(L))
        text_h = h[:, input_embeds.shape[1]:]
        logits = model.cosmos.head(text_h)
        cot_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), reasoning.reshape(-1))

        # ② Expert：flow matching
        model.condition = text_h
        x1 = target_actions[mode]
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)
        t = torch.rand(batch)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_target = x1 - x0
        v_pred = model.step_fn(x_t, t)
        fm_loss = F.mse_loss(v_pred, v_target)

        loss = cot_loss + fm_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"iter {it:4d}  cot_loss={cot_loss.item():.4f}  fm_loss={fm_loss.item():.4f}")


if __name__ == "__main__":
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = KAPPA
    target[1, :, 1] = -KAPPA
    target[2, :, 1] = 0.0

    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)

    print("\n训练后：历史 + 图片一起给，模型生成推理 + 预测轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj, text_ids = model.sample(HISTORIES[i:i + 1], IMAGES[i:i + 1])
            words = [list(VOCAB.keys())[list(VOCAB.values()).index(t)] for t in text_ids[0].tolist()]
            end = traj[0, -1, :2]
            print(f"  {name:8s}  推理={' '.join(words)}  终点(x,y)=({end[0]:6.2f}, {end[1]:6.2f})")
