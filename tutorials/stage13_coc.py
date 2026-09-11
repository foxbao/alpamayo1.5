"""Stage 13: 自回归 CoC 生成 —— 迷你 Cosmos-Reason Backbone

⚠️ 关键区分：本 stage 的文本是【模型生成的推理】，**不是**输入指令。

    stage10 的文本:  输入（外部给定的指令，如 "turn left in 10m"）
    stage13 的文本:  输出（模型看完图后【自己写出来】的推理）
                     如 "shift left due to curve" —— 模仿真实 CoC 的「动作 + due to + 原因」

    真实 Alpamayo 里两者都存在且角色不同：
        导航指令（输入）→ 影响推理怎么写；CoC 推理（输出）→ 其隐状态影响轨迹怎么出

本 stage 还演示【变长生成】：三条推理链长度不同（6 / 6 / 3），生成时遇到 <eos> 就停。
变长带来的两个机制：
    ① padding —— 训练时把短链补齐，用 IGNORE_INDEX 标记 pad（同训练教程 train2 的标签掩码）
    ② 生成时按 EOS 停止，而不是固定步数

注意：本 stage 把输入简化成了「只有图片」，省略了历史轨迹。完整版见 stage14_complete.py。
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
IGNORE_INDEX = -100  # 和真实代码一致（sft_base_model.py:35）

VOCAB = ["<bos>", "<eos>", "<pad>",
         "shift", "left", "right", "hold", "course", "due", "to", "curve"]
V = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BOS_ID, EOS_ID, PAD_ID = V["<bos>"], V["<eos>"], V["<pad>"]

# 三条 CoC 推理链，长度【不同】（真实 CoC 本来就是变长的）
CHAINS = [
    [V["shift"], V["left"], V["due"], V["to"], V["curve"], EOS_ID],    # 6 个
    [V["shift"], V["right"], V["due"], V["to"], V["curve"], EOS_ID],   # 6 个
    [V["hold"], V["course"], EOS_ID],                                  # 3 个 ← 更短
]
MAX_LEN = max(len(c) for c in CHAINS)

# 补齐成张量：输入用 <pad>，标签用 IGNORE_INDEX
REASONING_IN = torch.full((3, MAX_LEN), PAD_ID, dtype=torch.long)
REASONING_LAB = torch.full((3, MAX_LEN), IGNORE_INDEX, dtype=torch.long)
for _i, _c in enumerate(CHAINS):
    REASONING_IN[_i, : len(_c)] = torch.tensor(_c)
    REASONING_LAB[_i, : len(_c)] = torch.tensor(_c)


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
        """给定 [<bos>, r1, r2, ...]（已补齐到定长），返回推理部分的隐状态（去掉 <bos>）。

        训练和推理都走这一条路径，保证 condition 的长度和含义一致。
        """
        L = visual.shape[1] + text_in.shape[1]
        h = self.cosmos(visual, text_in, make_causal_mask(L))
        return h[:, visual.shape[1] + 1 :]                    # 去掉 <bos>

    def generate(self, image, max_len=MAX_LEN):
        """自回归生成推理：遇到 <eos> 就停（变长）。"""
        B = image.shape[0]
        visual = self.vit(image).last_hidden_state            # (B,17,64)
        text_ids = torch.full((B, 1), BOS_ID, dtype=torch.long)  # 从 <bos> 开始
        for _ in range(max_len):
            L = VISUAL_TOKENS + text_ids.shape[1]
            h = self.cosmos(visual, text_ids, make_causal_mask(L))
            next_tok = self.cosmos.head(h[:, -1]).argmax(-1, keepdim=True)  # 贪心取最大
            text_ids = torch.cat([text_ids, next_tok], dim=1)
            if bool((next_tok == EOS_ID).all()):
                break                                        # ← 变长：见到 <eos> 停

        # 把生成结果补齐到定长，使 condition 的形状与训练一致
        gen = text_ids[:, 1:]                                # 去掉 <bos>
        if gen.shape[1] < max_len:
            pad = torch.full((B, max_len - gen.shape[1]), PAD_ID, dtype=torch.long)
            gen = torch.cat([gen, pad], dim=1)
        text_in = torch.cat([torch.full((B, 1), BOS_ID, dtype=torch.long), gen], dim=1)
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
        r_in = REASONING_IN[mode]                         # (B,MAX) 用 <pad> 补齐
        r_lab = REASONING_LAB[mode]                       # (B,MAX) 用 IGNORE_INDEX 补齐

        # ① CosmosReason 训练：teacher forcing
        visual = model.vit(img).last_hidden_state         # (B,17,64)
        bos = torch.full((batch, 1), BOS_ID, dtype=torch.long)
        text_in = torch.cat([bos, r_in[:, :-1]], dim=1)        # (B,MAX)
        L = VISUAL_TOKENS + text_in.shape[1]
        h = model.cosmos(visual, text_in, make_causal_mask(L))
        text_h = h[:, VISUAL_TOKENS:]                     # (B,MAX)
        logits = model.cosmos.head(text_h)
        # ← 变长的代价：pad 位置用 ignore_index 跳过（同训练教程 train2 的标签掩码）
        cot_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            r_lab.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

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

    print("三条推理链（长度【不同】）：")
    for i, c in enumerate(CHAINS):
        words = " ".join(VOCAB[t] for t in c)
        print(f"  {['left','right','straight'][i]:9s} {len(c)} 个 token:  {words}")
    print(f"  → 最长 {MAX_LEN}，训练时短的补齐、用 IGNORE_INDEX 不算 loss\n")

    model = MiniVLA()
    # lr=5e-4：left/right 推理链共享前缀，地形较陡，1e-3 在部分种子上会崩
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    train(model, opt, target)
    model.eval()

    print("\n训练后：给一张图，模型自回归【生成】推理（见到 <eos> 停）+ 预测轨迹")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj, text_ids = model.sample(IMAGES[i:i + 1])
            words = [VOCAB[t] for t in text_ids[0].tolist()]
            n_gen = len(text_ids[0]) - 1                  # 去掉 <bos>
            end = traj[0, -1, :2]
            print(f"  {name:9s} 生成 {n_gen} 个 token: {' '.join(words):<40} 终点(x,y)=({end[0]:6.2f}, {end[1]:6.2f})")
