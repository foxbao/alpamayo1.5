"""real4: 轨迹几何 round-trip —— 连续 vs 离散，误差各有多大？

用【真实的】动作空间和 tokenizer（从 release config 实例化）做两件事：
  ① 连续 round-trip：轨迹 → (加速度,曲率) → 轨迹      ← 测最小二乘反解精度
  ② 量化 round-trip：轨迹 → tokens → 轨迹             ← 再加量化误差
并对比：量化到底损失了多少？

对应真实代码：
  action_space/unicycle_accel_curvature.py:234  traj_to_action（带正则的最小二乘反解）
  action_space/unicycle_accel_curvature.py:307  action_to_traj（积分）
  action_space/discrete_action_space.py:47      DiscreteTrajectoryTokenizer.encode
纯 CPU，秒级。
"""

import json
import glob
import os

import numpy as np
import torch
import hydra.utils as hyu

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000


def sec(t):
    print(f"\n{'='*64}\n{t}\n{'='*64}")


def load_cfg():
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/*/config.json"
    )
    with open(glob.glob(pattern)[0]) as f:
        return json.load(f)


def err(a, b):
    """逐 waypoint 的 L2 误差 (..., T, 3) -> (T,)"""
    return torch.linalg.norm(a - b, dim=-1).squeeze(0)


if __name__ == "__main__":
    cfg = load_cfg()
    action_space = hyu.instantiate(cfg["action_space_cfg"])
    tokenizer = hyu.instantiate(cfg["traj_tokenizer_cfg"])

    sec("① 配置（来自 release）")
    print(f"  动作空间: {type(action_space).__name__}")
    print(f"    n_waypoints={action_space.n_waypoints}, dt={action_space.dt}")
    print(f"    accel_std={action_space.accel_std.item():.4f}, "
          f"curvature_std={action_space.curvature_std.item():.4f}")
    print(f"  tokenizer: {type(tokenizer).__name__}")
    print(f"    num_bins={tokenizer.num_bins}, dims=[{tokenizer.dims_min}, {tokenizer.dims_max}]")
    print(f"    未来 token 数 = {action_space.n_waypoints} × 2 = "
          f"{action_space.n_waypoints * 2}")

    # 量化分辨率（归一化单位 → 物理单位）
    res_norm = (tokenizer.dims_max[0] - tokenizer.dims_min[0]) / (tokenizer.num_bins - 1)
    print(f"\n  量化分辨率（归一化）: {res_norm:.6f}")
    print(f"    → 加速度:  {res_norm * action_space.accel_std.item():.6f} m/s²")
    print(f"    → 曲率:    {res_norm * action_space.curvature_std.item():.8f} 1/m")

    # ---- 真实轨迹 ----
    sec("② 加载真实轨迹")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    hist_xyz = data["ego_history_xyz"].squeeze(0)      # (1, 16, 3)
    hist_rot = data["ego_history_rot"].squeeze(0)      # (1, 16, 3, 3)
    fut_xyz = data["ego_future_xyz"].squeeze(0)        # (1, 64, 3)
    fut_rot = data["ego_future_rot"].squeeze(0)
    print(f"  历史 {tuple(hist_xyz.shape)}，未来 {tuple(fut_xyz.shape)}")

    # ---- ① 连续 round-trip ----
    sec("③ 连续 round-trip：轨迹 → 动作 → 轨迹")
    action = action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
    fut_xyz_rt, _ = action_space.action_to_traj(action, hist_xyz, hist_rot)
    e_cont = err(fut_xyz_rt, fut_xyz)
    print(f"  动作 shape = {tuple(action.shape)}   （64 waypoints × (accel, curvature)）")
    print(f"  动作范围: accel [{action[...,0].min():.2f}, {action[...,0].max():.2f}], "
          f"curv [{action[...,1].min():.2f}, {action[...,1].max():.2f}]  (归一化单位)")
    print(f"\n  round-trip 误差（逐 waypoint L2, 单位 m）:")
    print(f"    mean = {e_cont.mean():.4f}   max = {e_cont.max():.4f}")

    # ---- ② 量化 round-trip ----
    sec("④ 量化 round-trip：轨迹 → tokens → 轨迹")
    tokens = tokenizer.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
    print(f"  tokens shape = {tuple(tokens.shape)}   （64×2 = 128）")
    print(f"  tokens 范围 = [{tokens.min().item()}, {tokens.max().item()}]  "
          f"（num_bins={tokenizer.num_bins}）")
    fut_xyz_q, _, _ = tokenizer.decode(hist_xyz, hist_rot, tokens)
    e_quant = err(fut_xyz_q, fut_xyz)
    print(f"\n  量化 round-trip 误差（逐 waypoint L2, 单位 m）:")
    print(f"    mean = {e_quant.mean():.4f}   max = {e_quant.max():.4f}")

    # ---- ③ 对比 ----
    sec("⑤ 结论：量化额外损失了多少？")
    print(f"  连续 round-trip  mean = {e_cont.mean():.4f} m")
    print(f"  量化 round-trip  mean = {e_quant.mean():.4f} m")
    print(f"  → 量化额外引入 {e_quant.mean() - e_cont.mean():.4f} m（"
          f"{(e_quant.mean()/e_cont.mean() - 1)*100:+.0f}%）")

    print(f"\n  终点误差：")
    print(f"    真值终点   = ({fut_xyz[0,-1,0]:.3f}, {fut_xyz[0,-1,1]:.3f})")
    print(f"    连续还原   = ({fut_xyz_rt[0,-1,0]:.3f}, {fut_xyz_rt[0,-1,1]:.3f})")
    print(f"    量化还原   = ({fut_xyz_q[0,-1,0]:.3f}, {fut_xyz_q[0,-1,1]:.3f})")

    sec("⑥ 前 8 个 waypoint 逐点对比（单位 m）")
    print(f"  {'#':>3}  {'真值 x':>8} {'真值 y':>8}  {'连续还原':>16}  {'量化还原':>16}  {'量化误差':>9}")
    for i in range(8):
        gx, gy = fut_xyz[0, i, 0].item(), fut_xyz[0, i, 1].item()
        cx, cy = fut_xyz_rt[0, i, 0].item(), fut_xyz_rt[0, i, 1].item()
        qx, qy = fut_xyz_q[0, i, 0].item(), fut_xyz_q[0, i, 1].item()
        print(f"  {i:>3}  {gx:>8.3f} {gy:>8.3f}  "
              f"({cx:>6.3f},{cy:>6.3f})  ({qx:>6.3f},{qy:>6.3f})  {e_quant[i]:>7.4f}")
