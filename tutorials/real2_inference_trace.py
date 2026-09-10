"""real2: 真实推理链路 trace（后半段：tokens → VLM → diffusion → 轨迹）

接 real1（前半段：数据 → prompt → tokens）。这一段需要加载 10B 模型（GPU）。

不重新实现——调用真实的 `sample_trajectories_from_data_with_vlm_rollout`，
并用轻量插桩（monkey-patch）打印中间步骤。

对应真实代码：
  models/alpamayo1_5.py:218  sample_trajectories_from_data_with_vlm_rollout
  models/base_model.py:172   fuse_traj_tokens
  models/alpamayo1_5.py:307  _find_eos_offset

运行（注意避开被 NoMachine 占用的 GPU 0）：
  CUDA_VISIBLE_DEVICES=1 python tutorials/real2_inference_trace.py
"""

import numpy as np
import torch

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000


def sec(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")


if __name__ == "__main__":
    sec("① 加载模型（10B，来自本地 HF 缓存）")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量 = {n_params/1e9:.2f} B")
    print(f"  vlm = {type(model.vlm).__name__}   expert = {type(model.expert).__name__}")
    print(f"  action_space = {type(model.action_space).__name__}")
    print(f"  diffusion = {type(model.diffusion).__name__}")

    sec("② 数据 → prompt → tokens（同 real1，此处略）")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
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
    input_ids = model_inputs["tokenized_data"]["input_ids"]
    print(f"  input_ids {tuple(input_ids.shape)}")

    sec("③ 历史轨迹 token 化（fuse_traj_tokens）")
    marker_id = model.config.traj_token_ids["history"]
    pos = (input_ids[0] == marker_id).nonzero().flatten()
    print(f"  占位符 <|traj_history|> id={marker_id}，共 {len(pos)} 个")
    print(f"  占据位置 {pos[0].item()} ~ {pos[-1].item()}")
    before = input_ids[0, pos[0] : pos[0] + 8].tolist()

    fused = model.fuse_traj_tokens(
        input_ids.clone(),
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )
    after = fused[0, pos[0] : pos[0] + 8].tolist()
    print(f"\n  替换前（前 8 个）: {before}   ← 全是占位符 id")
    print(f"  替换后（前 8 个）: {after}   ← 真实历史 token id")
    print("  （原始 input_ids 未被改，fuse 返回的是副本）")
    model_inputs["tokenized_data"]["input_ids"] = fused

    sec("④ 完整推理（VLM 生成 CoC → diffusion 采样 → 轨迹）")

    # 轻量插桩：包一层 diffusion.sample 和 _find_eos_offset
    _orig_sample = model.diffusion.sample

    def logged_sample(*args, **kwargs):
        print(f"    [diffusion.sample] batch_size={kwargs.get('batch_size')}, "
              f"n_steps={model.diffusion.num_inference_steps}")
        out = _orig_sample(*args, **kwargs)
        print(f"    [diffusion.sample] → 采样动作 shape = {tuple(out.shape)}")
        return out

    model.diffusion.sample = logged_sample

    _orig_find = model._find_eos_offset

    def logged_find(sequences, eos_token_id, device, warn=True):
        out = _orig_find(sequences, eos_token_id, device, warn=warn)
        print(f"    [_find_eos_offset] <|traj_future_start|> 后一位 = {out.tolist()}")
        return out

    model._find_eos_offset = staticmethod(logged_find)

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    sec("⑤ 输出")
    print(f"  pred_xyz {tuple(pred_xyz.shape)}   （B, num_traj_sets, num_traj_samples, T, 3）")
    print(f"  pred_rot {tuple(pred_rot.shape)}")
    print(f"\n  Chain-of-Causation:\n    {extra['cot'][0]}")

    pred = pred_xyz[0, 0, 0].float().cpu().numpy()          # (64, 3)
    gt = data["ego_future_xyz"][0, 0].numpy()               # (64, 3)
    end = pred[-1, :2]
    print(f"\n  预测终点 (x,y) = ({end[0]:.2f}, {end[1]:.2f}) m")
    print(f"  真值终点 (x,y) = ({gt[-1, 0]:.2f}, {gt[-1, 1]:.2f}) m")

    min_ade = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean()
    print(f"  minADE = {min_ade:.3f} m")
