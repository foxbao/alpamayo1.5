"""real6: 输入消融实验 —— 模型到底在看什么？

固定住随机种子（保证采样噪声一致），只改【一个】输入维度，看输出怎么变：
  ① baseline
  ② 帧序倒转（每相机内 3,2,1,0）      → 模型真用了时序信息吗？
  ③ 左右相机标签互换                   → 靠文字标签还是图像内容？
  ④ 历史轨迹置零（假装静止）           → 历史有多重要？
  ⑤ 只用 2 个相机                      → 相机冗余度

对应真实代码：helper.create_message / load_physical_aiavdataset
运行：CUDA_VISIBLE_DEVICES=1 python tutorials/real6_ablation.py
"""

import numpy as np
import torch

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000


def sec(t):
    print(f"\n{'='*68}\n{t}\n{'='*68}")


def run(model, processor, frames, cam_idx, hist_xyz, hist_rot):
    """跑一次推理。固定种子，保证差异只来自输入。"""
    messages = helper.create_message(frames=frames, camera_indices=cam_idx)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {"tokenized_data": inputs, "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot},
        "cuda",
    )
    torch.cuda.manual_seed_all(42)          # ← 关键：固定采样噪声
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256, return_extra=True,
        )
    cot = np.asarray(extra["cot"]).reshape(-1)          # 展平，取第一条
    return pred_xyz[0, 0, 0].float().cpu().numpy(), str(cot[0]) if cot.size else ""


if __name__ == "__main__":
    sec("① 加载模型 + 数据")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    frames_base = data["image_frames"].flatten(0, 1)      # (16, 3, H, W)
    cam_base = data["camera_indices"]                     # [0,1,2,6]
    hist_xyz = data["ego_history_xyz"]
    hist_rot = data["ego_history_rot"]
    print(f"  frames {tuple(frames_base.shape)}  camera_indices {cam_base.tolist()}")

    # ---- 消融变体 ----
    frames_rev = torch.flip(data["image_frames"], dims=[1]).flatten(0, 1)   # 每相机内倒序
    cam_swap = torch.tensor([2, 1, 0, 6])                          # 左右互换
    hist_zero = torch.zeros_like(hist_xyz)
    rot_eye = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(
        1, 1, hist_rot.shape[2], 1, 1
    ).to(hist_rot.dtype)

    # 2 相机数据（从缓存加载）
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    data2 = load_physical_aiavdataset(
        CLIP_ID, t0_us=T0_US, avdi=avdi,
        camera_features=[
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ],
    )
    frames2 = data2["image_frames"].flatten(0, 1)

    variants = [
        ("baseline",        frames_base, cam_base, hist_xyz,  hist_rot),
        ("帧序倒转",        frames_rev,  cam_base, hist_xyz,  hist_rot),
        ("左右相机标签互换", frames_base, cam_swap, hist_xyz,  hist_rot),
        ("历史轨迹置零",     frames_base, cam_base, hist_zero, rot_eye),
        ("只用 2 个相机",    frames2,     data2["camera_indices"], hist_xyz, hist_rot),
    ]

    sec("② 逐个跑消融（固定同一随机种子）")
    results = {}
    for name, fr, ci, hx, hr in variants:
        print(f"  跑 {name} ...", flush=True)
        pred, cot = run(model, processor, fr, ci, hx, hr)
        results[name] = (pred, cot)
        print(f"    终点 (x,y) = ({pred[-1,0]:7.2f}, {pred[-1,1]:7.2f})   CoC: {cot[:60]}")

    # ---- 对比 ----
    sec("③ 与 baseline 的差异")
    base_pred, base_cot = results["baseline"]
    print(f"  {'变体':<20} {'终点距离(m)':>12} {'平均轨迹距离(m)':>16}  CoC 是否相同")
    print("  " + "-" * 66)
    for name, (pred, cot) in results.items():
        if name == "baseline":
            print(f"  {name:<20} {'—':>12} {'—':>16}  —")
            continue
        end_d = np.linalg.norm(pred[-1, :2] - base_pred[-1, :2])
        mean_d = np.linalg.norm(pred[:, :2] - base_pred[:, :2], axis=-1).mean()
        same = "相同" if cot.strip() == base_cot.strip() else "**不同**"
        print(f"  {name:<20} {end_d:>12.2f} {mean_d:>16.2f}  {same}")

    sec("④ 各变体的 CoC（推理链）")
    for name, (_, cot) in results.items():
        print(f"\n  [{name}]")
        print(f"    {cot}")
