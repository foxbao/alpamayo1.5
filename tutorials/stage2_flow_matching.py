"""Stage 2: 向量场采样直觉 —— 在已知的 oracle 场上跑欧拉积分

【目的】把「采样器」和「模型」拆开看，先只看采样器：
  - 采样循环本身只是 `x = x + dt * step_fn(x, t)` —— 一个 for 循环，与神经网络无关
  - 从标准高斯噪声出发，被向量场一步步「推」向数据（这里被推向 target=0）
  - `step_fn` 才是模型该做的事（stage3~6 才补上），本 stage 用 target-x 顶替
  - temperature 缩放初始噪声：调小 → 更稳定但多样性更低（效果需实测，不保证更好）

【简化】这里的 `target - x` 是【人为给的收缩场】，不是训练出来的 flow matching：
  - 它是 dx/dt = -x 的解析方向，走 10 步后仍残留约 e^-1 的初始误差，且路径不直
  - 真正的 FM 速度是 v = x1 - x0（直线插值路径），且必须【训出来】——见 stage6
  - 所以本 stage 只回答「采样循环长什么样」，不回答「模型怎么学到这个场」
  - 真实 FlowMatching 还支持 CFG、任意 batch、可配积分步数/方法；这里是固定 batch 的最小版
"""

import torch

BATCH = 2
N_WAYPOINTS = 64
ACTION_DIM = 2
N_STEPS = 10

class FlowMatching:
    """流匹配：不从噪声一步到位，而是逐步"去噪"。

    循环：
    ① 从高斯噪声出发
    ② 问 step_fn：现在该往哪走？得到向量场 v
    ③ 沿 v 走一小步
    """
        
    def __init__(self, n_steps=N_STEPS):
        self.n_steps = n_steps

    def sample(self, step_fn, temperature=1.0):
        # ① 起点：标准高斯噪声（temperature 缩放噪声 → 控制多样性）
        x = torch.randn(BATCH, N_WAYPOINTS, ACTION_DIM) * temperature
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = torch.full((BATCH,), i / self.n_steps)   # 当前"时间"
            v = step_fn(x, t)                             # ② 向量场（网络预测）
            x = x + dt * v                                # ③ 欧拉一步
        return x

if __name__ == "__main__":
    fm=FlowMatching()
    
    target=torch.zeros(BATCH, N_WAYPOINTS, ACTION_DIM)
    
    def mock_step_fn(x, t):
        # 仅演示收缩场 dx/dt=target-x；t=1 时仍保留约 e^-1 的初始误差，
        # 并不是 stage6 的直线 flow matching 真值速度 x1-x0。
        return target-x

    x=torch.randn(BATCH, N_WAYPOINTS, ACTION_DIM)
    print("采样过程（重点看 std 怎么缩小）：")
    print(f"  start : mean={x.mean():+.3f}  std={x.std():.3f}")
    dt = 1.0 / N_STEPS
    
    for i in range(N_STEPS):
        t = torch.full((BATCH,), i / N_STEPS)
        x=x+dt*mock_step_fn(x, t)
        if i in (1,4,9):
            print(f"  step{i+1:2d}: mean={x.mean():+.3f}  std={x.std():.3f}")
    sampled=fm.sample(mock_step_fn)
    print(f"\n另一次独立采样: mean={sampled.mean():+.3f}  std={sampled.std():.3f}")
