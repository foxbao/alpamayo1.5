# 本机运行备忘（8× RTX 4090）

在这台机器（inspur-NF5468M6，8× RTX 4090，无头/SSH 环境）跑 Alpamayo 1.5 踩过的坑和验证过的解法。换机器或换网络后部分条目可能不适用。

## 1. 网络：下载走国内镜像（直连 huggingface.co / pypi.org 会超时）

本机代理 `127.0.0.1:7890` 对 huggingface.co 吞吐很慢（~74KB/s），且 dataset 的
`repo_info` 响应达 6MB 会直接 `Read timed out`。下载一律走国内镜像：

```bash
# pip 用清华镜像
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>

# Hugging Face 用 hf-mirror，且绕开慢代理
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

## 2. flash-attn 从源码编译（只编 sm_80，约快 4 倍）

4090 = compute 8.9 (sm_89)，但官方默认编 80/90/100/120 四个架构，其中三个用不上。
sm_80 的 cubin 对 sm_89 二进制兼容（同主版本 8，89≥80），所以只需编 sm_80：

```bash
export FLASH_ATTN_CUDA_ARCHS="80"   # 只编 sm_80；填 "89" 无效（代码只认 80/90/100/120 字面值）
export MAX_JOBS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE
uv pip install --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn==2.8.3
```

## 3. GPU / 显存

- **GPU 0 被 NoMachine（远程桌面 `nxnode.bin`）占 ~423MB**，跑 24GB 临界的大模型会 OOM。
  用 `CUDA_VISIBLE_DEVICES=1` 切到空闲卡（GPU 1~7 都空着）。
- **单张 4090 (24GB) 只能跑 `num_traj_samples=1`**（峰值 ~23GB）。16 采样要 ~40GB，
  需 80GB 卡或多卡。官方示例默认 16，本地要改成 1。
- 显存碎片导致的 OOM 可加：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 4. 数据集预下载（避免 streaming 联网超时）

`physical_ai_av` 默认按需 streaming 从 HF 拉 chunk 文件，会触发 `repo_info` 超时。
用仓库里的 `download_clip_data.py` 预下载到缓存：

```bash
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python download_clip_data.py <clip_id1> <clip_id2> ...
```

下载后进 `~/.cache/huggingface/hub/`，推理时直接读缓存、不再联网。

## 5. 无头跑 notebook（SSH 无浏览器）

```bash
cd notebooks    # 重要：notebook 用相对路径 clip_ids.parquet，必须在 notebooks/ 目录下执行
source ../a1_5_venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  jupyter nbconvert --to html --execute <notebook>.ipynb --output <name>.html
```

生成的 HTML 内嵌所有图和输出，scp 回本地浏览器打开即可。需要先 `uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nbconvert`。

## 6. 一键跑通模板

```bash
source a1_5_venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/alpamayo1_5/test_inference.py
```

## 其他

- 各 notebook 用的 clip：`inference.ipynb` / `inference_cam_num.ipynb` / `inference_vqa.ipynb`
  用 `clip_ids[774]` = `030c760c-ae38-49aa-9ad8-f5650a545d26`；`inference_nav.ipynb` 用
  `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4` 和 `c9c045a3-ebe9-4569-9ce3-a44068cf2e3b`。
- 官方示例都显式 `attn_implementation="sdpa"`（模型默认 `flash_attention_2`），
  所以跑这些示例不依赖 flash-attn；只有自己写推理、不传该参数时才会用到 flash-attn。
