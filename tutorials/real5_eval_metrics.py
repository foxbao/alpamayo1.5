"""real5: 评估指标 —— 别只看 minADE

用真实的模型 + 真实数据，算一整套指标：
  ① 精度：ADE / FDE / 终点误差 / 航向误差
  ② 舒适性：加速度 / 曲率 / jerk 的超限比例（用真实动作空间算）
  ③ 多样性：多次采样的差异（可选，需要更多显存）

对应真实代码：
  metrics/distance_metrics.py  DistanceMetrics
  metrics/metric_api.py        ReasoningSampler
运行：CUDA_VISIBLE_DEVICES=1 python tutorials/real5_eval_metrics.py
"""

import json
import glob
import os

import numpy as np
import torch
import hydra.utils as hyu

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
    p = os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/*/config.json"
    )
    with open(glob.glob(p)[0]) as f:
        return json.load(f)


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    cfg = load_cfg()
    action_space = hyu.instantiate(cfg["action_space_cfg"])

    sec("① 加载模型 + 数据 + 推理")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
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
    with torch.autocast("cuda", dtype=torch.bfloat16):
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
    sec("③ 舒适性指标（用真实动作空间算）")
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
              f"   (界限 ±9.8)")
        print(f"    曲率    mean={np.abs(curv).mean():.5f} max={np.abs(curv).max():.5f} 1/m"
              f"  (界限 ±0.33)")
        print(f"    jerk    mean={np.abs(jerk).mean():.3f}  max={np.abs(jerk).max():.3f} m/s³")
        return accel, curv, jerk

    acc_p, cur_p, jerk_p = comfort(pred[best], pred_r[best], "模型预测（最优样本）")
    comfort(gt_xyz, gt_rot, "真值")

    # ---------------- ④ 多样性 ----------------
    if pred.shape[0] > 1:
        sec("④ 多样性（多采样之间的差异）")
        pair = []
        for i in range(pred.shape[0]):
            for j in range(i + 1, pred.shape[0]):
                pair.append(np.linalg.norm(pred[i][:, :2] - pred[j][:, :2], axis=-1))
        pair = np.array(pair)
        print(f"  {pred.shape[0]} 条轨迹，{len(pair)} 对")
        print(f"  两两平均距离: mean={pair.mean():.3f}  max={pair.max():.3f} m")
        print(f"  终点两两距离: mean={np.linalg.norm(pred[:, -1, :2] - pred[:, -1, :2].mean(0), axis=-1).mean():.3f} m")
        print(f"\n  → 多样性越大 = 模型表达了越多的可能性（但太大会显得不确定）")

        sec("⑤ 关键对比：只看 minADE 会骗人")
        print(f"  minADE = {ades.min():.3f} m   ← 只看这个，模型「很好」")
        print(f"  但 {pred.shape[0]} 条采样里：最好的 {ades.min():.3f}，最差的 {ades.max():.3f}"
              f"（相差 {ades.max()/ades.min():.1f} 倍）")
        print(f"  离散程度（ADE 标准差）= {ades.std():.3f} m")
        print(f"\n  → minADE 只反映「最幸运的一次」；")
        print(f"    真实部署要看：分布的均值/方差、是否覆盖真值、以及舒适性")


if __name__ == "__main__":
    main()
