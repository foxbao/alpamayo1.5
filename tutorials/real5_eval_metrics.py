"""real5: 评估指标 —— 别只看 minADE

【目的】用真实的模型 + 真实数据算一整套指标，核心是学会「**怎么report结果**」：
    ① 精度：ADE / FDE / 终点误差 / 航向误差
    ② 运动学代理指标：反解的加速度 / 曲率 / jerk 统计
    ③ 多样性：多次采样的差异（可选，需要更多显存）
  三条要记住的结论：
    - **minADE 会骗人**：它是「K 条里挑最好」。同一条 clip 上 min 0.225 vs mean 0.616
      差 3 倍、max 1.006 差 4.5 倍 —— 真实部署必须看分布，不能只看 min
    - **位置误差小 ≠ 驾驶风格对**：历史运行里预测加速度峰值明显高于真值
      （1.423 vs 0.524 m/s²），所以舒适性指标必须和 ADE/FDE 一起报告
    - 此例 mean FDE 约为 mean ADE 的 5 倍，只说明终点误差高于全程平均，
      **不证明**每一步误差单调增长

【简化 / 结论边界】
  - 反解会**平滑动作、裁剪曲率**，所以不能据此验证「原始控制量是否超限」；
    动作空间的 bounds 也不是舒适性阈值
  - README 里的数值是**一次历史运行记录**（n=1、2 条采样），重跑会因采样和环境而变
  - 24GB 卡的容量取决于可用显存/输入长度/attention 后端；历史运行中 4 条曾 OOM，
    不是通用上限

【前提】需要 GPU + 约 22 GB 权重 + HF 权限。
  运行：N_SAMPLES=1 python tutorials/real5_eval_metrics.py（按机器选择可用 GPU）

对应真实代码：action_space/unicycle_accel_curvature.py；位移指标在本脚本中计算。
"""

import json
import os

import numpy as np
import torch
import hydra.utils as hyu
from huggingface_hub import hf_hub_download

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from alpamayo1_5.geometry.rotation import so3_to_yaw_torch

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
N_SAMPLES = int(os.environ.get("N_SAMPLES", "2"))   # 采样数；24GB 卡建议 ≤2


def sec(t):
    print(f"\n{'='*64}\n{t}\n{'='*64}")


def load_cfg():
    path = hf_hub_download("nvidia/Alpamayo-1.5-10B", "config.json")
    with open(path) as f:
        return json.load(f)


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pairwise_trajectory_distances(pred_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-timestep and endpoint distances for every unordered sample pair.

    Args:
        pred_xy: Predicted trajectories with shape ``(K, T, 2)``.

    Returns:
        Per-timestep distances with shape ``(K * (K - 1) / 2, T)`` and
        the corresponding endpoint distances with shape ``(K * (K - 1) / 2,)``.
    """
    if pred_xy.ndim != 3 or pred_xy.shape[-1] != 2 or pred_xy.shape[1] == 0:
        raise ValueError("pred_xy must have shape (K, T, 2) with T > 0")

    pairs = [
        np.linalg.norm(pred_xy[i] - pred_xy[j], axis=-1)
        for i in range(pred_xy.shape[0])
        for j in range(i + 1, pred_xy.shape[0])
    ]
    pairwise = np.asarray(pairs).reshape(-1, pred_xy.shape[1])
    return pairwise, pairwise[:, -1]


def main():
    if N_SAMPLES < 1:
        raise ValueError("N_SAMPLES must be positive")
    cfg = load_cfg()
    action_space = hyu.instantiate(cfg["action_space_cfg"])

    sec("① 加载模型 + 数据 + 推理")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    torch.cuda.manual_seed_all(42)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=N_SAMPLES,
            max_generation_length=256,
            return_extra=True,
        )
    print(f"  pred_xyz {tuple(pred_xyz.shape)}  (B, ns, nj, T, 3)  nj={N_SAMPLES}")

    pred = pred_xyz[0, 0].float().cpu().numpy()               # (nj, 64, 3)
    pred_r = pred_rot[0, 0].float().cpu().numpy()             # (nj, 64, 3, 3)
    gt_xyz = data["ego_future_xyz"][0, 0].numpy()             # (64, 3)
    gt_rot = data["ego_future_rot"][0, 0].numpy()             # (64, 3, 3)

    # ---------------- ② 精度指标 ----------------
    sec("② 精度指标")
    # 逐样本 ADE / FDE，取最好的一条（minADE）
    ades, fdes = [], []
    for k in range(pred.shape[0]):
        d = np.linalg.norm(pred[k][:, :2] - gt_xyz[:, :2], axis=-1)   # (64,)
        ades.append(d.mean())
        fdes.append(d[-1])
    ades, fdes = np.array(ades), np.array(fdes)
    best = int(ades.argmin())
    print(f"  样本数 nj = {pred.shape[0]}")
    print(f"  ADE  (平均位移误差): min={ades.min():.3f}  mean={ades.mean():.3f}  max={ades.max():.3f} m")
    print(f"  FDE  (终点位移误差): min={fdes.min():.3f}  mean={fdes.mean():.3f}  max={fdes.max():.3f} m")
    print(f"  minADE = {ades.min():.3f} m   ← 我们一直在看的就是这个")
    print(f"  minFDE = {fdes.min():.3f} m")

    end_err = np.linalg.norm(pred[best][-1, :2] - gt_xyz[-1, :2])
    print(f"\n  最优样本的终点误差 = {end_err:.3f} m")

    # 航向误差
    yaw_pred = so3_to_yaw_torch(torch.from_numpy(pred_r[best].astype(np.float32))).numpy()
    yaw_gt = so3_to_yaw_torch(torch.from_numpy(gt_rot.astype(np.float32))).numpy()
    yaw_err = np.abs(wrap_pi(yaw_pred - yaw_gt))
    print(f"  航向误差: mean={np.degrees(yaw_err.mean()):.2f}°  max={np.degrees(yaw_err.max()):.2f}°")

    # ---------------- ③ 舒适性指标 ----------------
    sec("③ 运动学代理指标（用真实动作空间反解）")
    print("  反解包含平滑和曲率裁剪；下列统计不是原始控制量，也不是舒适性认证。")
    hist_xyz = data["ego_history_xyz"].squeeze(0)
    hist_rot = data["ego_history_rot"].squeeze(0)

    def comfort(traj_xyz, rot_mats, name):
        fut = torch.from_numpy(traj_xyz[None].astype(np.float32))          # (1,64,3)
        fr = torch.from_numpy(rot_mats[None].astype(np.float32))           # (1,64,3,3)
        act = action_space.traj_to_action(hist_xyz, hist_rot, fut, fr)     # (1,64,2) 归一化
        accel = (act[..., 0] * action_space.accel_std + action_space.accel_mean).squeeze(0).numpy()
        curv = (act[..., 1] * action_space.curvature_std + action_space.curvature_mean).squeeze(0).numpy()
        jerk = np.diff(accel) / action_space.dt
        print(f"\n  [{name}]")
        print(f"    加速度  mean={np.abs(accel).mean():.3f}  max={np.abs(accel).max():.3f} m/s²"
              f"   (动作空间边界 {action_space.accel_bounds})")
        print(f"    曲率    mean={np.abs(curv).mean():.5f} max={np.abs(curv).max():.5f} 1/m"
              f"  (反解裁剪边界 {action_space.curvature_bounds})")
        print(f"    jerk    mean={np.abs(jerk).mean():.3f}  max={np.abs(jerk).max():.3f} m/s³")
        return accel, curv, jerk

    for k in range(pred.shape[0]):
        comfort(pred[k], pred_r[k], f"模型预测（样本 {k}）")
    comfort(gt_xyz, gt_rot, "真值")

    # ---------------- ④ 多样性 ----------------
    if pred.shape[0] > 1:
        sec("④ 多样性（多采样之间的差异）")
        pair, endpoint_pairwise = pairwise_trajectory_distances(pred[:, :, :2])
        print(f"  {pred.shape[0]} 条轨迹，{len(pair)} 对")
        pair_means = pair.mean(axis=1)
        print(f"  两两平均轨迹距离: mean={pair_means.mean():.3f}  max={pair_means.max():.3f} m")
        print(
            f"  终点两两距离: mean={endpoint_pairwise.mean():.3f} "
            f"max={endpoint_pairwise.max():.3f} m"
        )
        print("\n  → 距离只描述样本离散程度，不证明不同轨迹合理或概率校准良好。")

        sec("⑤ 关键对比：只看 minADE 会骗人")
        print(f"  minADE = {ades.min():.3f} m   ← 只看这个，模型「很好」")
        print(f"  但 {pred.shape[0]} 条采样里：最好的 {ades.min():.3f}，最差的 {ades.max():.3f}")
        print(f"  离散程度（ADE 标准差）= {ades.std():.3f} m")
        print("\n  → minADE 只反映「最幸运的一次」；")
        print("    真实部署要看：分布的均值/方差、是否覆盖真值、以及舒适性")


if __name__ == "__main__":
    main()
