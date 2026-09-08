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


# # Alpamayo 1.5: Navigation-Conditioned Trajectory Prediction

# Alpamayo 1.5 introduces the ability to **condition trajectory predictions on natural language navigation instructions**. Given a driving scene and an instruction like *"Turn right in 30m"*, the model produces trajectories that follow the specified route intention.
# 
# This notebook demonstrates:
# 1. **Basic trajectory prediction** with chain-of-causation reasoning
# 2. **Navigation-conditioned sampling** -- how the same scene yields different trajectory distributions depending on the navigation instruction
# 3. **Manual control** -- how to integrate navigation conditioning into your own pipeline
# 
# We use data from the NVIDIA [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

# In[2]:


import json
import mediapy as mp
import matplotlib.pyplot as plt

import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper, nav_utils
from alpamayo1_5.viz_utils import make_camera_grid, plot_bev_comparison


# ## Part 1: Basic Trajectory Prediction
# 
# First, we load the model and run standard inference -- the model observes camera images and ego history, then generates a chain-of-causation (CoC) reasoning trace followed by a predicted trajectory.

# In[3]:


model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
processor = helper.get_processor(model.tokenizer)


# We load a scene at an intersection in Sweden where the vehicle approaches a decision point -- it could turn right or continue straight.

# In[4]:


clip_id = "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4"
t0_us = 4_000_000

data_scene1 = load_physical_aiavdataset(clip_id, t0_us=t0_us)
# mp.show_images(data_scene1["image_frames"].flatten(0, 1).permute(0, 2, 3, 1), columns=4, width=200)   # 输入帧预览：脚本模式已注释


# In[5]:


messages = helper.create_message(
    data_scene1["image_frames"].flatten(0, 1),
    camera_indices=data_scene1["camera_indices"],
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = helper.to_device(
    {
        "tokenized_data": inputs,
        "ego_history_xyz": data_scene1["ego_history_xyz"],
        "ego_history_rot": data_scene1["ego_history_rot"],
    },
    "cuda",
)

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

print("Chain-of-Causation:\n", extra["cot"][0])


# In[6]:


for i in range(pred_xyz.shape[2]):
    pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
    gt_xy = data_scene1["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    plt.plot(pred_xy[0], pred_xy[1], "o-", label=f"Predicted #{i + 1}")
plt.plot(gt_xy[0], gt_xy[1], "r-", label="Ground Truth")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend(loc="best")
plt.axis("equal")
plt.title("Basic Trajectory Prediction (no navigation instruction)")
plt.savefig(f'nav_01.png', dpi=100, bbox_inches='tight')
plt.close()


# ## Part 2: Navigation-Conditioned Trajectory Sampling
# 
# One of the key new capabilities of Alpamayo 1.5: by providing a **navigation instruction**, you can steer the model's trajectory distribution.
# 
# We demonstrate this with a three-condition comparison:
# - **p(traj | nav)** (blue): conditioned on the actual navigation instruction
# - **p(traj)** (red): no navigation instruction (baseline)
# - **p(traj | opposite nav)** (green): conditioned on the *opposite* direction (counterfactual)
# - **GT** (black): ground truth trajectory
# 
# The same model and inference API are used for all three -- only the input text prompt changes.
# 
# Note: You can use diffusion temperature < 1.0 to get more stable but less diversity trajectory samples.

# In[7]:


nav_text = "Turn right in 30m"

print(f"Navigation instruction: {nav_text}")
print(f"Counterfactual (swapped): {nav_utils.swap_direction(nav_text)}")


# In[8]:


torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    nav_result = nav_utils.compare_nav_conditions(
        model=model,
        processor=processor,
        data=data_scene1,
        nav_text=nav_text,
        num_traj_samples=1,
        top_p=0.98,
        temperature=0.6,
        max_generation_length=256,
        return_extra=True,
        additional_nav_inference_kwargs={
            "diffusion_kwargs": {
                # The temperature for controlling the initial noise. Note that using
                # temperature < 1.0 will result in a more stable sampling with less diversity.
                "temperature": 0.6,
            }
        },
    )

print(f"Trajectories per condition: {nav_result.pred_with_nav.shape[2]}")


# In[9]:


camera_grid = make_camera_grid(
    data_scene1["image_frames"], camera_indices=data_scene1["camera_indices"]
)

fig = plot_bev_comparison(
    pred_with_nav=nav_result.pred_with_nav,
    pred_no_nav=nav_result.pred_no_nav,
    pred_counterfactual=nav_result.pred_counterfactual,
    nav_text=nav_result.nav_text,
    nav_text_swapped=nav_result.nav_text_swapped,
    gt_future_xyz=data_scene1.get("ego_future_xyz"),
    camera_images=camera_grid,
    title=f'Navigation: "{nav_text}"',
)
plt.savefig(f'nav_02.png', dpi=100, bbox_inches='tight')
plt.close()


# ### Classifier-Free Guidance (CFG) for Navigation-Conditioned Inference
# 
# Optionally, we can apply **classifier-free guidance** ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)) to amplify the effect of the navigation instruction on trajectory prediction. The `inference_guidance_weight` parameter ($\alpha$) relates to $w$ in Eq. (6) of the paper by $\alpha = 1 + w$:
# 
# - **$\alpha = 1$**: no guidance amplification (default conditioned prediction).
# - **$\alpha > 1$**: amplified guidance -- the model more aggressively follows the navigation instruction at the expense of sample diversity.
# - **$\alpha = 0$**: pure unconditioned prediction (ignores the navigation instruction entirely).
# 
# NOTE: To run this inference with CFG, you will need 60GB+ GPU memory.

# In[10]:


torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    nav_result = nav_utils.compare_nav_conditions(
        model=model,
        processor=processor,
        data=data_scene1,
        nav_text=nav_text,
        num_traj_samples=1,
        top_p=0.98,
        temperature=0.6,
        max_generation_length=256,
        return_extra=False,
        # comment out the following lines to use the default inference function
        nav_inference_fn=model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav,
        additional_nav_inference_kwargs={
            "diffusion_kwargs": {
                "use_classifier_free_guidance": True,
                # 0 = unguided only, 1 = guidance only, >1 = more guidance
                "inference_guidance_weight": 1.5,
                # The temperature for controlling the initial noise. Note that using
                # temperature < 1.0 will result in a more stable sampling with less diversity.
                "temperature": 0.6,
            }
        },
    )

print(f"Trajectories per condition: {nav_result.pred_with_nav.shape[2]}")


# In[11]:


camera_grid = make_camera_grid(
    data_scene1["image_frames"], camera_indices=data_scene1["camera_indices"]
)

fig = plot_bev_comparison(
    pred_with_nav=nav_result.pred_with_nav,
    pred_no_nav=nav_result.pred_no_nav,
    pred_counterfactual=nav_result.pred_counterfactual,
    nav_text=nav_result.nav_text,
    nav_text_swapped=nav_result.nav_text_swapped,
    gt_future_xyz=data_scene1.get("ego_future_xyz"),
    camera_images=camera_grid,
    title=f'Navigation: "{nav_text}"',
)
plt.savefig(f'nav_03.png', dpi=100, bbox_inches='tight')
plt.close()


# ### Try different instructions
# 
# The file `nav_demo_samples.json` contains examples of navigation instructions for clips in the PhysicalAI-AV dataset. You can pick any entry and see how the model responds.

# In[12]:


with open("nav_demo_samples.json") as f:
    nav_samples = json.load(f)

print(f"{len(nav_samples)} clips with navigation instructions:")
for s in nav_samples[:5]:
    print(f"  {s['nav_maneuver']:20s} | dist={s['distance_m']:5.1f}m | {s['nav_text']}")


# ## Part 3: Manual Navigation Instruction Overwrite
# 
# For researchers who want fine-grained control over the navigation instructions fed to the model, you can build conditioned inputs directly using `helper.create_message` with a `nav_text` argument and call the same inference API.
# 
# This lets you integrate navigation conditioning into your own pipeline -- swap in any text instruction and observe how the model's reasoning and trajectories change.
# 
# ### Specify manuevers

# In[13]:


frames = data_scene1["image_frames"].flatten(0, 1)

nav_text = "Turn left in 30m"
messages_nav = helper.create_message(
    frames,
    camera_indices=data_scene1["camera_indices"],
    nav_text=nav_text,
)

inputs_nav = processor.apply_chat_template(
    messages_nav,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

model_inputs_nav = helper.to_device(
    {
        "tokenized_data": inputs_nav,
        "ego_history_xyz": data_scene1["ego_history_xyz"],
        "ego_history_rot": data_scene1["ego_history_rot"],
    },
    "cuda",
)

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz_nav, pred_rot_nav, extra_nav = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs_nav,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

print(f"Navigation instruction: {nav_text}")
print(f"Chain-of-Causation (nav-conditioned):\n{extra_nav['cot'][0]}")

gt_xy = data_scene1["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
for i in range(pred_xyz_nav.shape[2]):
    pred_xy_i = pred_xyz_nav.cpu()[0, 0, i, :, :2].T.numpy()
    plt.plot(pred_xy_i[0], pred_xy_i[1], "o-", label=f"Predicted #{i + 1}", alpha=0.1)
plt.plot(gt_xy[0], gt_xy[1], "r-", label="Ground Truth")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.title("Nav-Conditioned Trajectory Prediction")
plt.savefig("nav_04_manual_nav.png", dpi=100, bbox_inches="tight")
plt.close()


# ### Specify distance from maneuver
# 
# You can also specify a distance in the provided, overwrite navigation instruction. Pay attention to how the model predicts trajectories conditioned on specified distance in navigation instructions.

# In[14]:


clip_id = "c9c045a3-ebe9-4569-9ce3-a44068cf2e3b"
t0_us = 2_000_000

data_scene2 = load_physical_aiavdataset(clip_id, t0_us=t0_us)
# mp.show_images(data_scene2["image_frames"].flatten(0, 1).permute(0, 2, 3, 1), columns=4, width=200)   # 输入帧预览：脚本模式已注释


# In[15]:


frames = data_scene2["image_frames"].flatten(0, 1)

nav_texts = [
    "Turn right in 9m",
    "Turn right in 18m",
    "Turn right in 36m",
]

for nav_text in nav_texts:
    messages_nav = helper.create_message(
        frames,
        camera_indices=data_scene2["camera_indices"],
        nav_text=nav_text,
    )
    inputs_nav = processor.apply_chat_template(
        messages_nav,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs_nav = helper.to_device(
        {
            "tokenized_data": inputs_nav,
            "ego_history_xyz": data_scene2["ego_history_xyz"],
            "ego_history_rot": data_scene2["ego_history_rot"],
        },
        "cuda",
    )

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz_nav, pred_rot_nav, extra_nav = (
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs_nav,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        )

    print(f"Navigation instruction: {nav_text}")
    print(f"Chain-of-Causation (nav-conditioned):\n{extra_nav['cot'][0]}")

    fig, axs = plt.subplots(1)
    gt_xy = data_scene2["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    for i in range(pred_xyz_nav.shape[2]):
        pred_xy_i = pred_xyz_nav.cpu()[0, 0, i, :, :2].T.numpy()
        axs.plot(pred_xy_i[0], pred_xy_i[1], "o-", label=f"Predicted #{i + 1}", alpha=0.1)
    axs.plot(gt_xy[0], gt_xy[1], "r-", label="Ground Truth")
    axs.set_xlabel("x (m)")
    axs.set_ylabel("y (m)")
    axs.set_xlim(-10, 30)
    axs.set_ylim(-25, 15)
    axs.legend(loc="best")
    axs.set_title(f"Nav-Conditioned Trajectory Prediction\n{nav_text}")
    plt.savefig(f'nav_05_{nav_text.replace(" ", "_")}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("-" * 80)

