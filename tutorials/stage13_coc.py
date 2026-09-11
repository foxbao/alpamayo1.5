"""Stage 13: 自回归 CoC 生成 —— 迷你 Cosmos-Reason Backbone

⚠️ 关键区分：本 stage 的文本是【模型生成的推理】，**不是**输入指令。

    stage10 的文本:  输入（外部给定的指令，如 "turn left"）
    stage13 的文本:  输出（模型看完图后【自己写出来】的推理，如 "road curves leftward"）

    为了不让两者混淆，本 stage 特意用了**和 stage10 完全不同的词**：
        stage10（输入指令）:  turn / left / right / continue / straight
        stage13（生成推理）:  road / curves / leftward / rightward / runs / ahead

    真实 Alpamayo 里两者都存在且角色不同：
        导航指令（输入）→ 影响推理怎么写；CoC 推理（输出）→ 其隐状态影响轨迹怎么出

注意：本 stage 还把输入简化成了「只有图片」，省略了历史轨迹（真实 Alpamayo 的
历史轨迹是重要输入）。完整版（历史+图片一起进 CosmosReason）见 stage14_complete.py。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM,
    build_vit, make_causal_mask, CosmosReason,
)

KAPPA = 0.1
IMG_SIZE = 16
VISUAL_TOKENS = 17   # ViT 输出：1 CLS + 16 patch

# 迷你词表：BOS + EOS + 【CoC 推理词】
# 模仿真实 Alpamayo 的 CoC 模式 —— 「动作 + due to + 原因」，例如：
#   "move to the left lane due to construction blocking the right side of our lane"
VOCAB = ["<bos>", "<eos>", "shift", "left", "right", "hold", "course", "due", "to", "curve", "clear"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BOS_ID = V["<bos>"]
EOS_ID = V["<eos>"]

# 三条 CoC 推理链（5 个词 + eos）—— 是【模型该生成的】，不是喂进去的
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


IMAGES = torch.stack([make_image(0), make_image(1), make_image(2)])  # (3,1,16,16)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
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

    def _reasoning_hidden(self, visual, text_in):
        """给定 [<bos>, r1, r2, ...]，返回推理部分的隐状态（去掉 <bos> 位置）。

        训练和推理都走这一条路径，保证 condition 的长度和含义一致。
        """
        L = visual.shape[1] + text_in.shape[1]
        h = self.cosmos(visual, text_in, make_causal_mask(L))
        return h[:, visual.shape[1] + 1 :]                    # 去掉 <bos>

    def generate(self, image):
        """自回归生成推理，隐状态 = condition。"""
        B = image.shape[0]
        n_gen = REASONING.shape[1]                            # 生成几个 token
        visual = self.vit(image).last_hidden_state            # (B,17,64)
        text_ids = torch.full((B, 1), BOS_ID, dtype=torch.long)  # 从 <bos> 开始
        for _ in range(n_gen):
            L = VISUAL_TOKENS + text_ids.shape[1]
            h = self.cosmos(visual, text_ids, make_causal_mask(L))
            next_tok = self.cosmos.head(h[:, -1]).argmax(-1, keepdim=True)  # 贪心取最大
            text_ids = torch.cat([text_ids, next_tok], dim=1)
        # 与训练一致：用「去掉 <bos> 和末尾 <eos>」的推理前缀算 condition
        text_in = text_ids[:, :-1]                            # [<bos>, r1, r2, r3]
        self.condition = self._reasoning_hidden(visual, text_in)
        return text_ids

    def sample(self, image):
        text_ids = self.generate(image)
        action = self.fm.sample(self.step_fn, batch_size=image.shape[0])
        traj = self.action_space.action_to_traj(action)
        return traj, text_ids


def train(model, opt, target_actions, n_iters=5000, batch=64):
    model.train()
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        img = IMAGES[mode]
        reasoning = REASONING[mode]                       # (B,4) 目标推理（模型该生成的）

        # ① CosmosReason 训练：teacher forcing
        visual = model.vit(img).last_hidden_state         # (B,17,64)
        bos = torch.full((batch, 1), BOS_ID, dtype=torch.long)
        text_in = torch.cat([bos, reasoning[:, :-1]], dim=1)   # [bos, road, curves, leftward]
        L = VISUAL_TOKENS + text_in.shape[1]
        h = model.cosmos(visual, text_in, make_causal_mask(L))
        text_h = h[:, VISUAL_TOKENS:]                     # 用于语言建模损失
        logits = model.cosmos.head(text_h)
        cot_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), reasoning.reshape(-1))

        # ② Expert 训练：condition = 推理前缀隐状态（去 <bos>），与推理时一致
        model.condition = model._reasoning_hidden(visual, text_in)
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

    print("词表里的推理词（模型该【生成】的）:", [w for w in VOCAB if w not in ("<bos>", "<eos>")])
    print("（对比 stage10 的【输入】指令词: turn / left / right / continue / straight）\n")

    model = MiniVLA()
    # lr=5e-4：left/right 的推理链共享前两个词（road curves），损失地形比旧词表更陡，
    # lr=1e-3 在部分种子上会崩溃（三个场景生成同一句）。降到 5e-4 才稳定。
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    train(model, opt, target)
    model.eval()

    print("\n训练后：给一张图，模型自回归【生成】推理 + 预测轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj, text_ids = model.sample(IMAGES[i:i + 1])
            words = [VOCAB[t] for t in text_ids[0].tolist()]
            end = traj[0, -1, :2]
            print(f"  {name:8s}  生成的推理={' '.join(words)}  终点(x,y)=({end[0]:6.2f}, {end[1]:6.2f})")
