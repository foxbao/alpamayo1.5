"""Stage 2: 向量场采样直觉 —— 已知收缩场上的欧拉积分。

这里的 ``target - x`` 是人为提供的 oracle 向量场，用来展示采样循环；
并非训练得到的 flow-matching 网络。真正的插值、损失和训练见 stage6。
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
