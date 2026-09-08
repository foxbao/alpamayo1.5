#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 本机运行配置：HF 走 hf-mirror、绕开慢代理、用空闲 GPU 1（必须在 import 之前设置，否则不生效）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 避开被 NoMachine 占用的 GPU 0
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(key, None)


# # Alpamayo 1.5: Camera Number Ablation Demo

# This notebook demonstrates how the number of camera inputs affects Alpamayo 1.5 trajectory prediction. The same sample is run with 1, 2, and 4 cameras, and the resulting chain-of-thought reasoning and predicted trajectories are compared side by side.

# In[2]:


import mediapy as mp
import pandas as pd
import matplotlib.pyplot as plt

import torch
import physical_ai_av
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper


# ### Load model and construct data preprocessor

# In[3]:


model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
processor = helper.get_processor(model.tokenizer)


# ### Define camera configurations and load data

# In[4]:


clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
clip_id = clip_ids[774]

avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

camera_configs = {
    "1 cam (front wide)": [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
    ],
    "2 cam (front wide + front tele)": [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ],
    "4 cam (left + front wide + right + front tele)": [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ],
}

all_data = {}
all_model_inputs = {}

for config_name, cam_features in camera_configs.items():
    data = load_physical_aiavdataset(
        clip_id,
        avdi=avdi,
        camera_features=cam_features,
    )

    messages = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"[{config_name}] seq length: {inputs.input_ids.shape}")

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    all_data[config_name] = data
    all_model_inputs[config_name] = model_inputs


# ### Run inference for each camera configuration

# In[5]:


results = {}

for config_name, model_inputs in all_model_inputs.items():
    print(f"Running inference: {config_name}")

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

    results[config_name] = {
        "pred_xyz": pred_xyz.cpu(),
        "cot": extra["cot"][0],
    }


# ### Show camera frames

# In[6]:


# mp.show_images(data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1), columns=4, width=200)   # 输入帧预览：脚本模式已注释


# ### Compare the outputs from different camera inputs

# In[7]:


# compare the cot outputs
for config_name, res in results.items():
    print(f"Chain-of-Thought — {config_name}")
    print(res["cot"])
    print()

# compare the trajectory outputs
ref_data = list(all_data.values())[0]
gt_xy = ref_data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

plt.figure(figsize=(6, 6))
for idx, (config_name, res) in enumerate(results.items()):
    pred_xys = res["pred_xyz"][0, 0, :, :, :2].numpy()
    for pred_xy in pred_xys:
        plt.plot(
            pred_xy.T[0],
            pred_xy.T[1],
            "o-",
            color=colors[idx],
            markersize=3,
            alpha=0.25,
        )

plt.plot(gt_xy[0], gt_xy[1], "r-", linewidth=2, label="Ground Truth")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory Comparison Across Camera Configurations")
plt.legend(loc="best", fontsize=8)
plt.axis("equal")
plt.tight_layout()
plt.savefig(f'cam_num_01.png', dpi=100, bbox_inches='tight')
plt.close()


# In[ ]:




