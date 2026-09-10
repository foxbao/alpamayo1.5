import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, Expert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
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


def build_vit():
    """一个真实的 ViT（transformers），小自定义配置匹配 toy 尺寸。

    patch_size=4 → 16×16 图切成 4×4 = 16 个 patch token；
    hidden_size=64 → 和我们 HIDDEN 对齐；
    num_labels=0 → 不要分类头，只要特征。
    """
    cfg = ViTConfig(
        image_size=16, patch_size=4, num_channels=1,
        hidden_size=HIDDEN, num_hidden_layers=4, num_attention_heads=4,
        intermediate_size=128, num_labels=0,
    )
    return ViTModel(cfg)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = build_vit()   # 真 ViT（替代 toy CNN）
        self.in_proj = ActionInProj()
        self.expert = Expert()
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
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = KAPPA     # left
    target[1, :, 1] = -KAPPA    # right
    target[2, :, 1] = 0.0       # straight

    model = MiniVLA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)

    print("\n训练后：给一张图，模型读出方向并预测轨迹")
    with torch.no_grad():
        for name, m in [("left", 0), ("right", 1), ("straight", 2)]:
            img = IMAGES[m:m + 1]               # (1,1,16,16)
            traj = model.sample(img)
            end = traj[0, -1, :2]
            print(f"  {name:8s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
