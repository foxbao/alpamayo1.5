"""Stage 0: 骨架 —— 先把「模块接口 + 数据流」钉死，再往里填实现

【目的】建立 VLA 的整体心智模型，本 stage **不做任何真实计算**，只跑通形状：
    ① 条件先行：VLM(图 + 历史) → 条件隐状态 (B, 32, 64)
    ② 扩散采样：噪声 x ─in_proj→ Expert(读条件) ─out_proj→ 向量场 v ─dt·v→ x
    ③ 动作→轨迹：action (B,64,2) → traj (B,64,3)
  核心只有 `MiniVLA.sample` 的三行，以及每个张量的形状约定。
  重点记住「三个 64」：64 个 waypoint（时间轴）、HIDDEN=64（特征轴）——两者巧合同值。

【简化】6 个模块全是【空壳】，forward 一律返回 zeros/随机张量：
  - VLM 不读图、不读历史，直接返回 zeros(B, 32, 64)
  - Expert 不做 attention，ActionInProj/OutProj 也返回 zeros
  - ActionSpace.action_to_traj 返回全零轨迹（积分逻辑 stage1 才补）
  - FlowMatching 的 10 步循环是真的，但 step_fn 是假的，所以「采样」不出任何东西
  - batch / 序列长度 / 维度全是写死的全局常量，**只支持固定 batch**
  真实 release 是 8B VLM + 独立 expert 去噪器 + Unicycle 动作空间，见 README §六。
"""

import torch
import torch.nn as nn

# ---- 超参数（先定死）----
BATCH = 2  # batch
N_WAYPOINTS = 64  # 未来 64 个点（6.4s @ 10Hz）
ACTION_DIM = 2  # 动作 = (加速度, 曲率)
HIDDEN = 64  # 隐状态维度（先很小，方便观察）
VLM_SEQ_LEN = 32  # VLM 输出的"条件"序列长度
N_HIST = 16  # 历史轨迹离散 token 数
N_IMAGES = 16  # 4 相机 × 4 时间帧


class ActionSpace:
    def action_dim(self):
        return (N_WAYPOINTS, ACTION_DIM)

    def action_to_traj(self, action):  # (B,64,2) -> (B,64,3)
        return torch.zeros(BATCH, N_WAYPOINTS, 3)  # stage 1 再实现积分


# ---- 2) VLM：看图+历史，产出"条件"隐状态 ----
class VLM(nn.Module):
    def forward(self, images, history_tokens):
        # images: (B,16,3,H,W)  history_tokens: (B,16)
        # 返回的 (B, L, HIDDEN) 就是"条件"——VLA 的灵魂
        return torch.zeros(BATCH, VLM_SEQ_LEN, HIDDEN)


# ---- 3) 动作 -> embedding 投影 ----
class ActionInProj(nn.Module):
    def forward(self, x, t):  # x:(B,64,2) t:(B,) -> (B,64,HIDDEN)
        return torch.zeros(BATCH, N_WAYPOINTS, HIDDEN)


# ---- 4) embedding -> 向量场 投影 ----
class ActionOutProj(nn.Module):
    def forward(self, h):  # (B,64,HIDDEN) -> (B,64,2)
        return torch.zeros(BATCH, N_WAYPOINTS, ACTION_DIM)


# ---- 5) Expert：去噪器，cross-attn 读 VLM 隐状态（先空着）----
class CrossAttnExpert(nn.Module):
    def forward(self, action_embeds, vlm_kv):
        # action_embeds:(B,64,HIDDEN)  vlm_kv:(B,L,HIDDEN) -> (B,64,HIDDEN)
        return torch.zeros(BATCH, N_WAYPOINTS, HIDDEN)


# ---- 6) Flow Matching：欧拉积分采样 ----
class FlowMatching:
    def sample(self, step_fn, n_steps=10):
        x = torch.randn(BATCH, N_WAYPOINTS, ACTION_DIM)  # 从噪声出发
        for i in range(n_steps):
            t = torch.full((BATCH,), i / n_steps)
            x = x + (1 / n_steps) * step_fn(x, t)  # 欧拉一步
        return x

# ---- 顶层：把一切串起来 ----
class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = VLM()
        self.expert = CrossAttnExpert()
        self.in_proj = ActionInProj()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()

    def step_fn(self, x, t):                      # 扩散的每一步
        emb = self.in_proj(x, t)                  # 噪声动作 -> embedding
        h = self.expert(emb, self.vlm_kv)         # 去噪（读 VLM 条件）
        return self.out_proj(h)                   # -> 向量场 v

    def sample(self, images, history_tokens):
        self.vlm_kv = self.vlm(images, history_tokens)    # ① 先算条件
        action = self.fm.sample(self.step_fn)             # ② 扩散采样动作
        return self.action_space.action_to_traj(action)   # ③ 动作->轨迹

    
    
if __name__ == "__main__":
    print("=" * 55)
    print("Stage 0：张量维度自检")
    print("=" * 55)
    
    images = torch.randn(BATCH, N_IMAGES, 3, 224, 224)
    history_tokens = torch.randint(0, 1000, (BATCH, N_HIST))
    print(f"输入 images:          {tuple(images.shape)}")
    print(f"输入 history_tokens:  {tuple(history_tokens.shape)}")
    
    model = MiniVLA()

    vlm_kv=model.vlm(images, history_tokens)
    print(f"\n① VLM 条件(KV):      {tuple(vlm_kv.shape)}   ← 这就是'语言/视觉条件'")
    
    # ② 扩散一步的内部数据流
    x0=torch.randn(BATCH, N_WAYPOINTS, ACTION_DIM)
    t=torch.zeros(BATCH)
    emb=model.in_proj(x0, t)
    h=model.expert(emb, vlm_kv)
    v=model.out_proj(h)
    print(f"② 初始噪声动作:       {tuple(x0.shape)}")
    print(f"   action_in_proj:    {tuple(emb.shape)}")
    print(f"   expert 输出:       {tuple(h.shape)}")
    print(f"   向量场 v:          {tuple(v.shape)}")
    
    # ③ 完整采样
    model.vlm_kv = vlm_kv
    action=model.fm.sample(model.step_fn)
    traj=model.action_space.action_to_traj(action)
    print(f"\n③ 采样动作:           {tuple(action.shape)}")
    print(f"   最终轨迹:          {tuple(traj.shape)}")


