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


# # Alpamayo 1.5 Demo

# This notebook loads example data from the NVIDIA [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) and runs the Alpamayo 1.5 model, producing output trajectories and associated chain-of-causation reasoning traces.

# In[2]:


import numpy as np
import mediapy as mp
import pandas as pd

import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper


# ### Load model and construct data preprocessor

# In[3]:


model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
processor = helper.get_processor(model.tokenizer)


# ### Load and prepare data

# In[4]:


clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
clip_id = clip_ids[774]

data = load_physical_aiavdataset(clip_id)

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
print("seq length:", inputs.input_ids.shape)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")


# ### Model inference

# In[5]:


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

print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])


# ## Visualizing data and results

# In[6]:


# mp.show_images(data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1), columns=4, width=200)   # 输入帧预览：脚本模式已注释


# In[7]:


import matplotlib.pyplot as plt

for i in range(pred_xyz.shape[2]):
    pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    plt.plot(pred_xy[0], pred_xy[1], "o-", label=f"Predicted Trajectory #{i + 1}")
plt.plot(gt_xy[0], gt_xy[1], "r-", label="Ground Truth Trajectory")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend(loc="best")
plt.axis("equal")


# In[8]:


pred_xy_all = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=1).mean(-1)
print("minADE:", diff.min(), "meters")

