"""real6: 输入消融实验 —— 模型到底在看什么？

【目的】学会用【消融】而不是直觉去回答「哪个模态重要」，本脚本最重要的产出
  是一个**方法论**：把混淆的变量拆开。固定文本生成种子、并在 diffusion 入口
  重新设种子，使轨迹初始噪声与文本长度无关。每次只改一种输入：
    ① baseline
    ② 帧序倒转（每相机内 3,2,1,0）      → 模型真用了时序信息吗？
    ③ 左右相机标签互换                   → 靠文字标签还是图像内容？
    ④ **【隔离】只改 VLM 历史条件**      → 只置零 prompt 里的历史 token，
                                           ego_history_xyz/rot 保持真实，v0 不受影响
    ⑤ **【混淆】历史全置零**（连 v0 一起）→ 历史有【两条】进入模型的路径
    ⑥ 只用 2 个相机                      → 相机冗余度

  ★ 必须理解的分解（历史实验数值）：
        历史全置零的总影响  44.26 m
          ├─ VLM 条件部分    11.05 m (25%)   ← VLM 对历史的使用：中等
          └─ v0 部分        ~33 m   (75%)   ← 主导，但这是【积分机制】的效应
    **所以不能说「历史是 VLM 的绝对主力」**：大部分差异来自 v0 被置零
    （车从速度 0 开始积分，自然几乎不动），而不是 VLM 忽略了历史。

【简化 / 结论边界】**这些仍是 n=1 的探索性观察**，不能据此给出普遍的「模态影响力排序」：
  - 历史既进 VLM、又用于 `estimate_t0_states` 估 v0；置零会同时改两条路径，
    且属于偏离训练分布的干预
  - 一条 clip 上标签互换影响小，**不代表标签普遍无用**，也不能证明模型只靠图像辨认相机
  - CoC 是否相同只是文本观察，不能证明其推理忠实性
  - 历史版本只在整段推理开头固定种子；修正后才在 diffusion 入口再设种子，
    因此 README 里的旧数值不能当成严格配对消融结果
  要推广需要多 clip、多种子、配对噪声和误差统计。

【前提】需要 GPU + 约 22 GB 权重 + HF 权限，且要跑**多次**推理，耗时依环境而定。
  运行：python tutorials/real6_ablation.py（按机器选择可用 GPU）

对应真实代码：helper.create_message / load_physical_aiavdataset
"""

from contextlib import ExitStack
from functools import partial
from unittest.mock import patch

import numpy as np
import torch

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000


def sec(t):
    print(f"\n{'='*68}\n{t}\n{'='*68}")


def sample_with_fixed_seed(sample_fn, *args, seed=42, **kwargs):
    """Reset the RNG at the diffusion boundary, after variable-length VLM generation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return sample_fn(*args, **kwargs)


def run(model, processor, frames, cam_idx, hist_xyz, hist_rot, zero_prompt_history=False):
    """分别固定文本与动作采样种子；不保证跨硬件的位级复现。

    zero_prompt_history=True 时，只把【送进 VLM prompt 的历史 token】置零，
    而 `ego_history_xyz/rot` 仍保持真实——这样 `action_to_traj` 估计出的积分初速度
    v0 不受影响，从而把「VLM 条件」和「积分初速度」两个效应分离开。
    （历史同时用于这两条路径，不区分就无法把差异归因给 VLM。）
    """
    messages = helper.create_message(frames=frames, camera_indices=cam_idx)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {"tokenized_data": inputs, "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot},
        "cuda",
    )
    torch.manual_seed(42)
    fixed_sample = partial(sample_with_fixed_seed, model.diffusion.sample)
    ctxs = [
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        patch.object(model.diffusion, "sample", fixed_sample),
    ]
    if zero_prompt_history:
        orig_fuse = model.fuse_traj_tokens

        def zeroed_fuse(input_ids, traj_data=None):
            if traj_data is not None and traj_data.get("ego_history_xyz") is not None:
                rot = traj_data["ego_history_rot"]
                traj_data = {
                    "ego_history_xyz": torch.zeros_like(traj_data["ego_history_xyz"]),
                    "ego_history_rot": torch.eye(3).to(rot).expand_as(rot).clone(),
                }
            return orig_fuse(input_ids, traj_data)

        ctxs.append(patch.object(model, "fuse_traj_tokens", zeroed_fuse))

    with ExitStack() as stack:
        for c in ctxs:
            stack.enter_context(c)
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
    ).to("cuda").eval()
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

    # 直接切片同一批输入，避免再次加载引入不同帧。
    keep = (cam_base == 1) | (cam_base == 6)
    frames2 = data["image_frames"][keep].flatten(0, 1)

    variants = [
        # (名称, frames, cam_idx, hist_xyz, hist_rot, 只改VLM历史条件)
        ("baseline",             frames_base, cam_base, hist_xyz,  hist_rot, False),
        ("帧序倒转",             frames_rev,  cam_base, hist_xyz,  hist_rot, False),
        ("左右相机标签互换",      frames_base, cam_swap, hist_xyz,  hist_rot, False),
        ("【隔离】只改VLM历史条件", frames_base, cam_base, hist_xyz,  hist_rot, True),
        ("【混淆】历史全置零",     frames_base, cam_base, hist_zero, rot_eye,  False),
        ("只用 2 个相机",         frames2,     cam_base[keep], hist_xyz, hist_rot, False),
    ]

    sec("② 逐个跑消融（固定同一随机种子）")
    results = {}
    for name, fr, ci, hx, hr, zp in variants:
        print(f"  跑 {name} ...", flush=True)
        pred, cot = run(model, processor, fr, ci, hx, hr, zero_prompt_history=zp)
        results[name] = (pred, cot)
        print(f"    终点 (x,y) = ({pred[-1,0]:7.2f}, {pred[-1,1]:7.2f})   CoC: {cot[:60]}")

    # ---- 对比 ----
    sec("③ 与 baseline 的差异")
    print("  历史同时用于 VLM 条件和积分初始速度。")
    print("  →「【隔离】」只把送进 prompt 的历史 token 置零，v0 仍真实：得到【纯 VLM 条件】的影响。")
    print("  →「【混淆】」连 v0 一起置零：这一行是两种效应的叠加，不能单独归因给 VLM。")
    print("  本实验只有一条 clip；差异大小不能证明普遍的模态重要性或 CoC 忠实性。")
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
