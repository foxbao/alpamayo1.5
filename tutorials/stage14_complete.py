"""Stage 14: 完整输入 + CoC —— 历史 + 图片一起进 CosmosReason（toy 中最完整的输入侧示例）

对比 stage13：这里把历史轨迹编码与图片编码一起拼进 CosmosReason，
再自回归生成短文本。仍省略多相机、多帧、导航和真实 CoC 的语义监督。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM,
    build_vit, make_causal_mask, CosmosReason, HistoryEncoder,
)

KAPPA = 0.1
IMG_SIZE = 16
HIST_LEN = 8          # 历史步数
VISUAL_TOKENS = 17    # ViT 输出：1 CLS + 16 patch

# CoC 词表 + 推理链（同 stage13）
# ⚠️ 这些是模型【生成】的推理（描述"看到什么"），不是输入指令。
#    对比 stage10 的【输入】指令词：turn / left / right / continue / straight
VOCAB = ["<bos>", "<eos>", "shift", "left", "right", "hold", "course", "due", "to", "curve", "clear"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BOS_ID = V["<bos>"]
EOS_ID = V["<eos>"]
REASONING = torch.tensor([
    [V["shift"], V["left"], V["due"], V["to"], V["curve"], EOS_ID],    # "shift left due to curve"
    [V["shift"], V["right"], V["due"], V["to"], V["curve"], EOS_ID],   # "shift right due to curve"
    [V["hold"], V["course"], V["due"], V["to"], V["clear"], EOS_ID],   # "hold course due to clear"
])


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


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_enc = HistoryEncoder()
        self.vit = build_vit()
        self.cosmos = CosmosReason(VOCAB_SIZE)
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
        for _ in range(REASONING.shape[1]):                  # 生成 4 个 token
            L = input_embeds.shape[1] + text_ids.shape[1]
            h = self.cosmos(input_embeds, text_ids, make_causal_mask(L))
            next_tok = self.cosmos.head(h[:, -1]).argmax(-1, keepdim=True)
            text_ids = torch.cat([text_ids, next_tok], dim=1)

        # 固定长度教学例：去掉最后的第 3 个生成 token（期望为 EOS，但不保证）。
        text_in = text_ids[:, :-1]                           # [<bos>, t1, t2]
        self.condition = self._reasoning_hidden(input_embeds, text_in)   # (B, 2, 64)
        return text_ids

    def _reasoning_hidden(self, input_embeds, text_in):
        """给定 [<bos>, r1, r2, ...]，返回推理部分的隐状态（去掉 <bos> 位置）。"""
        L = input_embeds.shape[1] + text_in.shape[1]
        h = self.cosmos(input_embeds, text_in, make_causal_mask(L))
        return h[:, input_embeds.shape[1] + 1 :]             # 去掉 <bos>

    def sample(self, hist, img):
        text_ids = self.generate(hist, img)
        action = self.fm.sample(self.step_fn, batch_size=hist.shape[0])
        traj = self.action_space.action_to_traj(action)
        return traj, text_ids


def train(model, opt, target_actions, n_iters=5000, batch=64):
    model.train()
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
        text_h = h[:, input_embeds.shape[1]:]                # 用于语言建模损失
        logits = model.cosmos.head(text_h)
        cot_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), reasoning.reshape(-1))

        # ② Expert：condition = 推理前缀隐状态（去 <bos>），与推理时一致
        model.condition = model._reasoning_hidden(input_embeds, text_in)   # (B,2,64)
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
    torch.manual_seed(0)
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = KAPPA
    target[1, :, 1] = -KAPPA
    target[2, :, 1] = 0.0

    model = MiniVLA()
    # lr=5e-4：同 stage13，left/right 推理链共享前缀，地形更陡，1e-3 不稳定
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    train(model, opt, target)
    model.eval()

    print("\n训练后：历史 + 图片一起给，模型生成推理 + 预测轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj, text_ids = model.sample(HISTORIES[i:i + 1], IMAGES[i:i + 1])
            words = [VOCAB[t] for t in text_ids[0].tolist()]
            end = traj[0, -1, :2]
            print(f"  {name:8s}  推理={' '.join(words)}  终点(x,y)=({end[0]:6.2f}, {end[1]:6.2f})")
