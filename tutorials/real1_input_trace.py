"""real1: 真实推理链路 trace（前半段：数据 → 坐标 → prompt → tokens）

不重新实现任何东西——只把真实链路跑一遍，把每一步的中间结果打出来，
让你看清「真实的输入契约」到底长什么样。

对应真实代码：
  load_physical_aiavdataset.py:27  load_physical_aiavdataset
  helper.py:77                     create_message
纯 CPU，不需要加载 10B 模型。
"""

import torch

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000


def sec(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")


def _load_amayo_tokenizer():
    """复刻 base_model._build_processor：Qwen processor + 按【真实顺序】加 token。

    真实顺序（base_model.py:270-283）：
      1) 先加 traj_vocab_size 个离散轨迹 token <i0>..<i3999>   ← 漏了这步 id 会差 4000
      2) 再加 SPECIAL_TOKENS（add_special_tokens=True 时）
    """
    import json

    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor
    from alpamayo1_5.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN

    # 从 release 的 config.json 读 traj_vocab_size，保证和真实一致
    cfg_path = hf_hub_download("nvidia/Alpamayo-1.5-10B", "config.json")
    with open(cfg_path) as f:
        traj_vocab_size = json.load(f)["traj_vocab_size"]

    proc = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", min_pixels=163840, max_pixels=196608
    )
    tok = proc.tokenizer
    # 1) 离散轨迹 token（真实里就是这 4000 个）
    tok.add_tokens([f"<i{v}>" for v in range(traj_vocab_size)])
    # 2) 特殊 token（release 配置 add_special_tokens=True）
    tok.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    tok.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)
    tok.traj_token_ids = {k: tok.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()}
    return tok


if __name__ == "__main__":
    sec("① 加载数据（真实数据集，来自本地 HF 缓存）")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:22s} {str(tuple(v.shape)):<24} {v.dtype}")
        else:
            print(f"  {k:22s} {v!r}")

    sec("② 相机与时间帧")
    print(f"  相机索引 camera_indices = {data['camera_indices'].tolist()}")
    print("    （0=左前 1=前广 2=右前 6=前长焦）")
    print(f"  相对时间戳 relative_timestamps:\n{data['relative_timestamps']}")
    print(f"  → {data['image_frames'].shape[0]} 相机 × {data['image_frames'].shape[1]} 时间帧"
          f" × {tuple(data['image_frames'].shape[2:])}")

    sec("③ 坐标变换：世界坐标 → ego 局部坐标")
    hist_world = data["ego_history_xyz"][0, 0]           # (16, 3)
    fut_world = data["ego_future_xyz"][0, 0]             # (64, 3)
    print(f"  历史轨迹（ego 局部坐标, 前3步）:\n{hist_world[:3].numpy().round(3)}")
    print(f"  未来轨迹（ego 局部坐标, 前3步）:\n{fut_world[:3].numpy().round(3)}")
    print(f"\n  注意 t0 时刻的位置 = {hist_world[-1].numpy().round(4)}  ← 应该是全 0（ego 原点）")
    print("  （原代码把世界坐标用 R_t0^{-1} 旋转 + 平移到了 ego 坐标系）")

    sec("④ 构造 chat message（helper.create_message）")
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    print(f"  message 条数: {len(messages)}")
    for m in messages:
        role = m["role"]
        n_img = sum(1 for c in m["content"] if c["type"] == "image")
        texts = [c["text"] for c in m["content"] if c["type"] == "text"]
        print(f"\n  [{role}]  图片 {n_img} 张, 文本片段 {len(texts)} 段")
        if role == "user":
            print(f"    首段文本: {texts[0]!r}")
            print(f"    末段文本: {texts[-1]!r}")
        else:
            print(f"    文本: {texts[0]!r}")

    sec("⑤ tokenize（processor.apply_chat_template）")
    processor = helper.get_processor(_load_amayo_tokenizer())
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"  inputs 键: {list(inputs.keys())}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:22s} {tuple(v.shape)}  {v.dtype}")

    ids = inputs["input_ids"][0]
    print(f"\n  序列长度 = {len(ids)}")
    print(f"  前 12 个 token: {ids[:12].tolist()}")
    print(f"  后 12 个 token: {ids[-12:].tolist()}")

    sec("⑥ 拆解序列：图片 token vs 文本 token")
    from alpamayo1_5.models.base_model import SPECIAL_TOKENS

    grid = inputs["image_grid_thw"]                       # (16, 3)
    merge = getattr(processor.image_processor, "merge_size", 2)   # Qwen3-VL 空间合并
    n_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    n_visual = n_patches // (merge * merge)               # patch → visual token（÷ merge²）
    print(f"  每张图的 patch 网格 (t×h×w): {grid[0].tolist()}  (共 {len(grid)} 张)")
    print(f"  patch 总数 = {n_patches}，空间合并 {merge}×{merge} → 视觉 token = {n_visual}")
    print(f"  序列总长 = {len(ids)}  →  文本 token ≈ {len(ids) - n_visual}")
    print(f"  即：**图片占了约 {n_visual/len(ids):.0%} 的序列长度**")

    sec("⑦ 定位关键特殊 token")
    tok = processor.tokenizer
    for key in ("traj_history_start", "traj_history", "traj_history_end", "cot_start"):
        tokstr = SPECIAL_TOKENS[key]
        tid = tok.convert_tokens_to_ids(tokstr)
        pos = (ids == tid).nonzero().flatten().tolist()
        if len(pos) > 6:
            shown = f"{pos[:3]} ... {pos[-2:]}"
        else:
            shown = str(pos)
        print(f"  {key:22s} id={tid:<8} 出现 {len(pos):>3} 次  位置 {shown}")
