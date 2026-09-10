import torch

# ---- 超参数 ----
BATCH = 2
N_WAYPOINTS = 64
ACTION_DIM = 2    # (加速度, 曲率)
DT = 0.1          # 0.1s = 10Hz

# 归一化参数（动作是标准化过的，积分前要先反归一化）
ACCEL_MEAN, ACCEL_STD = 0.0, 1.0
KAPPA_MEAN, KAPPA_STD = 0.0, 1.0
KAPPA_BOUND = 0.33   # 曲率上限 → 最小转弯半径约 3m


class ActionSpace:
    """单轮车(unicycle)运动学模型。"""

    def action_dims(self):
        return (N_WAYPOINTS, ACTION_DIM)

    def action_to_traj(self, action, v0=5.0):
        """把动作积分成轨迹。

        action: (B, N, 2)  归一化的 (加速度, 曲率)
        v0:     初始速度标量（真实代码里从历史轨迹估计）
        返回:   (B, N, 3)  ego 坐标系下的 (x, y, z)
        """
        # 1) 反归一化
        accel = action[..., 0] * ACCEL_STD + ACCEL_MEAN   # (B, N)
        kappa = action[..., 1] * KAPPA_STD + KAPPA_MEAN   # (B, N)
        kappa = kappa.clamp(-KAPPA_BOUND, KAPPA_BOUND)

        # 2) 积分：从 ego 原点出发（车头朝 +x）
        v = torch.full_like(accel[:, :1], v0)     # (B, 1) 初始速度
        theta = torch.zeros_like(accel[:, :1])    # (B, 1) 初始航向
        x = torch.zeros_like(accel[:, :1])        # (B, 1) 起点
        y = torch.zeros_like(accel[:, :1])

        xs, ys = [], []
        for t in range(N_WAYPOINTS):
            a = accel[:, t:t + 1]              # (B, 1)
            k = kappa[:, t:t + 1]              # (B, 1)
            v = v + a * DT                     # ① 速度
            theta = theta + k * v * DT         # ② 航向（用新速度）
            x = x + v * torch.cos(theta) * DT  # ③ 位置
            y = y + v * torch.sin(theta) * DT
            xs.append(x)
            ys.append(y)

        x = torch.cat(xs, dim=-1)              # (B, N)
        y = torch.cat(ys, dim=-1)              # (B, N)
        z = torch.zeros_like(x)                # 平面假设
        return torch.stack([x, y, z], dim=-1)  # (B, N, 3)


if __name__ == "__main__":
    as_ = ActionSpace()

    # 测试 A：全零动作 → 匀速直线
    a0 = torch.zeros(BATCH, N_WAYPOINTS, ACTION_DIM)
    traj = as_.action_to_traj(a0)
    print("A) 全零动作（匀速直线）")
    print("   终点:", traj[0, -1].tolist(), "  y 应该≈0, x≈v0*N*dt =", 5.0 * N_WAYPOINTS * DT)

    # 测试 B：恒定曲率 → 转弯（接近圆弧）
    b0 = torch.zeros(BATCH, N_WAYPOINTS, ACTION_DIM)
    b0[..., 1] = 0.1                            # 每步曲率 0.1
    traj_b = as_.action_to_traj(b0)
    print("\nB) 恒定曲率 0.1（转弯）")
    print("   终点:", traj_b[0, -1].tolist(), "  → 应该转出去了（x 不再单调增）")

    # 测试 C：恒定加速度 → 直线加速
    c0 = torch.zeros(BATCH, N_WAYPOINTS, ACTION_DIM)
    c0[..., 0] = 1.0                            # 每步加速度 1
    traj_c = as_.action_to_traj(c0)
    print("\nC) 恒定加速度 1（直线加速）")
    print("   终点:", traj_c[0, -1].tolist(), "  → x 应该远超 32m（速度变快了）")

        
