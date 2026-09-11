"""Stage 6: 训练 —— 让 condition 真正控制轨迹（flow matching 训练）"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ActionSpace, FlowMatching, ActionInProj, ActionOutProj, CrossAttnExpert,
    N_WAYPOINTS, ACTION_DIM, HIDDEN, VLM_SEQ_LEN,
)


# ---- 可学习的 condition 生成器（stage6 新增，替代 stage5 的 mock_vlm）----
class ConditionGenerator(nn.Module):
    """把"指令"映射成 condition。真实里是 VLM，这里用可学习的 Embedding。"""
    def __init__(self, num_instructions, hidden=HIDDEN, seq_len=VLM_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(num_instructions, hidden)
        self.seq_len = seq_len

    def forward(self, idx):
        return self.embed(idx)[:, None, :].expand(-1, self.seq_len, -1)  # (B, L, HIDDEN)


# ---- MiniVLA ----
class MiniVLA(nn.Module):
    def __init__(self, num_instructions):
        super().__init__()
        self.cond_gen = ConditionGenerator(num_instructions)
        self.in_proj = ActionInProj()
        self.expert = CrossAttnExpert()
        self.out_proj = ActionOutProj()
        self.action_space = ActionSpace()
        self.fm = FlowMatching()
        self.condition = None

    def step_fn(self, x, t):
        emb = self.in_proj(x, t)
        h = self.expert(emb, self.condition)
        return self.out_proj(h)

    def sample(self, instr_idx):
        self.condition = self.cond_gen(instr_idx)
        action = self.fm.sample(self.step_fn, batch_size=instr_idx.shape[0])
        return self.action_space.action_to_traj(action)


# ---- 训练 ----
def train(model, opt, target_actions, n_iters=3000, batch=32):
    model.train()
    for it in range(n_iters):
        idx = torch.randint(0, target_actions.shape[0], (batch,))   # 随机指令
        model.condition = model.cond_gen(idx)                        # (B,L,H)
        x1 = target_actions[idx]                                     # (B,N,2) 目标动作
        x0 = torch.randn(batch, N_WAYPOINTS, ACTION_DIM)             # 噪声
        t = torch.rand(batch)                                        # 随机时间
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1    # 插值
        v_target = x1 - x0                                           # 真值向量场
        v_pred = model.step_fn(x_t, t)                               # 网络预测
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"iter {it:4d}  loss={loss.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    # 三个指令的目标动作（恒定曲率）
    target = torch.zeros(3, N_WAYPOINTS, ACTION_DIM)
    target[0, :, 1] = 0.05    # "left"   -> 正曲率（左转）
    target[1, :, 1] = -0.05   # "right"  -> 负曲率（右转）
    target[2, :, 1] = 0.0     # "straight"

    model = MiniVLA(3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, opt, target)
    model.eval()

    print("\n训练后采样（看终点 y 的符号）：")
    with torch.no_grad():
        for name, i in [("left", 0), ("right", 1), ("straight", 2)]:
            traj = model.sample(torch.tensor([i]))
            end = traj[0, -1, :2]
            print(f"  {name:8s}  终点(x,y) = ({end[0]:6.2f}, {end[1]:6.2f})")
