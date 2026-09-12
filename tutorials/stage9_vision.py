"""Stage 9: 视觉编码 —— 补上 VLA 的「V」，图 → visual tokens → condition

【目的】理解视觉是怎么进模型的：
  - 图 → ViT → (B, 1+16, HIDDEN)：**1 个 CLS token + 16 个 patch token**
    视觉不是被「压成一个向量」，而是展开成一条 token 序列 —— 这样才能
    和文本 / 历史 token 拼在一起进 transformer
  - 这 17 个 visual token 直接当 condition 喂给 Expert
  - 模型行为：给一张「车道线」图，读出方向并预测轨迹
  - 顺带看到 patch 化的代价：16×16 的图切成 4×4 的 patch 只剩 16 个 token

【简化】
  - 图片是 16×16 **单通道合成灰度图**（画一条斜/直线表示左弯/右弯/直行）；
    真实是 1920×1080 RGB 多相机，且受 min_pixels/max_pixels 约束
  - ViT 是 `ViTConfig` 随机初始化（4 层、hidden 64），**不下载预训练权重**；
    真实 Cosmos-Reason2 的视觉塔是预训练的，且和 LLM 联合训练
  - 只有 1 个相机、1 帧；真实是多相机（4~8 路）× 多帧（每路 4 帧）
  - 没有相机名文字标签（stage12 才加），也没有位置/外参输入
  - 训练数据只有 3 张固定图 → 3 条固定动作，是「记忆」而非泛化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM,
    build_vit,
)

IMG_SIZE = 16
IMG_CH = 1
KAPPA = 0.1


def make_image(mode):
    """单张 16×16 的"车道线"图。mode: 0=左弯 1=右弯 2=直行。"""
    img = torch.zeros(IMG_CH, IMG_SIZE, IMG_SIZE)
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
        self.vision = build_vit()   # 随机初始化的小型 ViT（使用 Transformers 实现）
        self.in_proj = ActionInProj()
        self.expert = CrossAttnExpert()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()
        self.condition = None

    def encode_image(self, image):
        # ViT 输出 (B, 1+16, HIDDEN) = 1 个 CLS token + 16 个 patch token
        return self.vision(image).last_hidden_state

    def step_fn(self, x, t):
        emb = self.in_proj(x, t)
        h = self.expert(emb, self.condition)
        return self.out_proj(h)

    def sample(self, image):
        self.condition = self.encode_image(image)     # (B, 17, HIDDEN)
        action = self.fm.sample(self.step_fn, batch_size=image.shape[0])
        return self.action_space.action_to_traj(action)


def train(model, opt, target_actions, n_iters=5000, batch=64):
    model.train()
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        imgs = IMAGES[mode]                          # (B,1,16,16)
        model.condition = model.encode_image(imgs)   # (B,17,HIDDEN)
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

    print("\n训练后：给一张图，模型读出方向并预测轨迹")
    with torch.no_grad():
        for name, m in [("left", 0), ("right", 1), ("straight", 2)]:
            img = IMAGES[m:m + 1]               # (1,1,16,16)
            traj = model.sample(img)
            end = traj[0, -1, :2]
            print(f"  {name:8s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
