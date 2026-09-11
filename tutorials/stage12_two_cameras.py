"""Stage 12: 双相机：共享 ViT，先融合每路标签与图片，再拼接条件。

裸 concat 后直接做 cross-attention 无法把相邻标签和图片绑定；这里用共享的
小型 transformer 在每路相机内建立联系。真实 VLM 在更长的完整序列上融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN,
    build_vit,
)

KAPPA = 0.1
IMG_SIZE = 16


def make_image(mode):
    """前视：竖/斜线（同 stage9）。mode: 0=左 1=右 2=直。"""
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


FRONT = torch.stack([make_image(0), make_image(1), make_image(2)])  # (3,1,16,16)
# 第二个相机的输入：把同一场景旋转 90°。
# ⚠️ 这仍是【合成】的，不是真实相机标定投影（toy 用 16×16 假图，做不了真投影）。
#    真实的多相机处理见 real1（16 张真实图 + 相机文字标签）和 real6（相机消融）。
#    这里用「旋转」而非「转置」：转置是镜像、翻转手性，物理上拍不出来；
#    旋转至少对应「相机装成另一个角度」。
SIDE = torch.rot90(FRONT, k=1, dims=(-2, -1))


# 相机文本标签：真实代码写 "Front camera" 这类短语，这里用两个词
VOCAB = {"front": 0, "side": 1}
CAMERA_LABELS = torch.tensor([[0], [1]])   # (2, 1)


class CameraLabelEncoder(nn.Module):
    """相机标签编码器（词 → 向量，一个词就一个 token）。"""
    def __init__(self, vocab_size=2, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)

    def forward(self, token_ids):
        return self.embed(token_ids)   # (B, L) -> (B, L, HIDDEN)


class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = build_vit()             # 共享 ViT（两个相机用同一个）
        self.text_enc = CameraLabelEncoder()   # 相机标签编码器
        self.pos_embed = nn.Parameter(torch.empty(1, 18, HIDDEN))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            HIDDEN, 4, dim_feedforward=HIDDEN * 2, dropout=0.0, batch_first=True
        )
        self.camera_fusion = nn.TransformerEncoder(layer, num_layers=1)
        self.in_proj = ActionInProj()
        self.expert = CrossAttnExpert()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()
        self.condition = None

    def encode(self, front_img, side_img):
        B = front_img.shape[0]
        f = self.vit(front_img).last_hidden_state    # (B,17,64)
        s = self.vit(side_img).last_hidden_state     # (B,17,64)

        # 相机身份 = 文本 token，拼在各自视觉 token 前面（不是"加向量"）
        front_label = self.text_enc(CAMERA_LABELS[0:1]).expand(B, -1, -1)  # (B,1,64)
        side_label = self.text_enc(CAMERA_LABELS[1:2]).expand(B, -1, -1)   # (B,1,64)

        f = torch.cat([front_label, f], dim=1)   # (B,18,64) = [front] + 17 视觉
        s = torch.cat([side_label, s], dim=1)    # (B,18,64) = [side]  + 17 视觉
        # 图片 token 在进入 Expert 前就读到了本路标签，不能省掉这一步。
        f = self.camera_fusion(f + self.pos_embed)
        s = self.camera_fusion(s + self.pos_embed)
        return torch.cat([f, s], dim=1)          # (B,36,64)

    def step_fn(self, x, t):
        emb = self.in_proj(x, t)
        h = self.expert(emb, self.condition)
        return self.out_proj(h)

    def sample(self, front_img, side_img):
        self.condition = self.encode(front_img, side_img)
        action = self.fm.sample(self.step_fn, batch_size=front_img.shape[0])
        return self.action_space.action_to_traj(action)


def train(model, opt, target_actions, n_iters=5000, batch=64):
    model.train()
    for it in range(n_iters):
        mode = torch.randint(0, 3, (batch,))
        model.condition = model.encode(FRONT[mode], SIDE[mode])   # (B,36,64)
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

    print("\n训练后：两张图（前视+侧视）+ 文本标签一起给")
    names = ["left", "right", "straight"]
    with torch.no_grad():
        for i, name in enumerate(names):
            traj = model.sample(FRONT[i:i + 1], SIDE[i:i + 1])
            end = traj[0, -1, :2]
            print(f"  {name:8s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
