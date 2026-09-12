"""Stage 12: 多相机 —— 共享 ViT + 相机身份标签（并列分支）

【目的】回答「多相机怎么区分」这个 VLA 的经典问题：
  - **共享 ViT**：两个相机用同一个 ViT（不是各一套权重），所以输出在同一空间里可比
  - 相机身份不用外参，而用**文本标签**（"front" / "side"）——真实代码写 "Front camera"
  - 标签是【文本 token 拼在各自视觉 token 前面】，而不是「加一个向量」到图上
  - 关键的一步：在每路内部用一个小 transformer 先让**视觉 token 读到本路标签**，
    再把两路 concat。**裸 concat 后直接做 cross-attention 无法把相邻标签和图片绑定**
    —— 标签和图片只是碰巧挨着，模型没有理由知道它们对应
  - 顺带一个结论：这条路不需要相机外参，靠「相机名 + 序列顺序」传递身份/布局线索

【简化】
  - 第二路相机是**合成**的：把前视图旋转 90°（真实的多相机是标定好的不同视角投影）
    用旋转而非转置，是因为转置会镜像翻转手性、物理上拍不出来
  - 16×16 灰度假图，做不了真实的相机标定投影
  - 每路只有 1 帧；真实每路 4 帧，用 `frame 0..3` 文字标签标注
  - 相机标签词表只有 2 个词（front/side），真实是短语（"Front camera" 等）
  - 只有 2 个相机；真实 4~8 路。没有历史、没有导航指令
  - 「标签无用论」不可从本 stage 推出：真实里标签是否必要需靠 real6 那类消融来验证
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
