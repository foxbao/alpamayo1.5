"""Stage 6: 训练 —— 让 condition 真正控制轨迹（flow matching 训练）

【目的】这是主线里第一个【会学习】的 stage，理解 FM 的训练目标本身：
  - 构造训练对：x_t = (1-t)·x0 + t·x1（x0 噪声、x1 数据），真值速度 v_target = x1 - x0
  - 损失就是最朴素的 MSE(v_pred, v_target)——没有噪声调度、没有后验采样
    （对比 DDPM 预测噪声 ε 并注入后验噪声，见 ddpm_1d.py）
  - 训练/推理必须走同一个 step_fn，形状对齐
  - 条件用 3 个可学习 Embedding 表示「左/右/直」，看采样终点 y 的符号是否听话
  - **训练/采样噪声必须对齐**：这里都在标准高斯下，所以采样 temperature=1.0

【简化】
  - condition 是 `nn.Embedding` 查表 + 广播到 (B,L,H)，没有任何真实输入：
    没有图、没有历史、没有文本；「指令」只是一个整数下标
  - 只有 3 条固定指令、目标动作是手工写死的恒定曲率 (0.05/-0.05/0)
  - 单步预测、无 CFG、无 EMA、无 lr 调度；3000 步能在 CPU 上跑完
  - 真实训练目标还包括 CoC 的语言建模损失（见 stage13 的 cot_loss + fm_loss）
  - Expert 仍是 cross-attention 版（真实为 prefix，见 stage4 / exp_prefix_expert.py）
  注意：单输出 MSE 只会拟合条件均值，无法表达多峰——见 stage8。
"""

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
