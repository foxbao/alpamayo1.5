# Repository Guidelines

## Scope and Layout

This repository contains the Alpamayo 1.5 inference release and a small,
CPU-only MiniVLA teaching track. The installable package is under
`src/alpamayo1_5/`:

- `models/` contains the VLM wrapper, trajectory tokenization, and action
  projections. `models/alpamayo1_5.py` owns the VLM rollout plus diffusion
  expert pipeline.
- `action_space/` contains trajectory/action representations and the unicycle
  acceleration-curvature kinematics. `diffusion/` contains samplers such as
  flow matching, and `geometry/` contains rotation and coordinate helpers.
- `helper.py` builds Qwen chat prompts and recursively moves tensors to a
  device. `load_physical_aiavdataset.py` adapts the gated PhysicalAI-AV
  dataset. `nav_utils.py` implements navigation comparisons and route removal.
- `test_inference.py` is a full inference example despite its name;
  `visualize_inference.py` is a command-line dashboard/video renderer and
  `viz_utils.py` contains plotting helpers.

`notebooks/` has standard, navigation, camera-count, and VQA examples. Each
notebook also has a checked-in `.py` script suitable for headless execution;
run those scripts from the repository root unless their documentation says
otherwise. `tutorials/stage0_framework.py` through `stage15_cfg.py` are
conceptual MiniVLA stages and run on CPU with toy data. `tutorials/real1_*`
through `real6_*` inspect or evaluate the real release and may require model
weights, a GPU, and cached dataset clips. `tutorials/common.py` is shared by
stages 6 through 15.

## Environment and Commands

Use Python 3.12 and `uv`; the project requires exactly Python 3.12 and pins
PyTorch 2.8.0 and Transformers 4.57.1:

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
uv tool run ruff check src
python -m pytest -q tests
```

`tests/test_tutorials.py` contains CPU-only regression and one-step training
checks for the tutorials. There is no CI configuration. Add focused tests under
`tests/test_<module>.py` for geometry, tokenization, action-space, diffusion,
or other pure behavior. Do not make ordinary tests download weights, access
the gated dataset, or require CUDA. A useful dependency-free smoke check is:

```bash
python -m compileall -q src tutorials tests
```

For the full example, authenticate with Hugging Face and run
`uv run python src/alpamayo1_5/test_inference.py`. It downloads roughly 22 GB
of model weights and accesses the gated PhysicalAI-AV dataset; this is not a
lightweight test. The default release model uses Flash Attention 2. When
Flash Attention cannot be built, install without it and pass
`attn_implementation="sdpa"` to `from_pretrained`, as shown in the examples.

The visualization entry point accepts a clip and output directory:

```bash
uv run python src/alpamayo1_5/visualize_inference.py \
  --clip-id <clip-id> --t0-us 5100000 \
  --output-dir outputs/alpamayo1_5_inference
```

It writes generated media under `outputs/`, which is ignored by git. Notebook
execution with `nbconvert` should be launched from `notebooks/` because some
examples use relative paths such as `clip_ids.parquet`.

Machine- and network-specific workarounds (HF mirror variables, proxy
settings, GPU selection, Flash Attention build flags, and 24 GB VRAM limits)
are documented in `RUN_NOTES.md`; do not copy those settings into library
code.

## Data, Shapes, and Runtime Assumptions

The dataset loader normally returns four cameras and four frames per camera:
`image_frames` has shape `(N_cameras, num_frames, 3, H, W)`, while
`camera_indices` has shape `(N_cameras,)`. Prompt construction commonly uses
`image_frames.flatten(0, 1)`. History and future trajectories are batched as
`(B, 1, T, 3)` with rotation matrices `(B, 1, T, 3, 3)`; the standard horizon
is 64 future waypoints at 10 Hz (6.4 seconds). Preserve these conventions when
extending the inference path and document any intentional shape change.

Model inference expects CUDA, BF16 autocast, Hugging Face authentication, and
downloaded or streamable gated data. A single 24 GB GPU generally supports
`num_traj_samples=1`; larger sample counts need more VRAM. Set seeds explicitly
when comparing navigation conditions or input ablations so sampling noise does
not obscure the input change.

## Coding and Documentation Style

Use four-space indentation, type annotations, `snake_case` for functions and
variables, and `PascalCase` for classes. Keep tensor shapes in docstrings or
short comments when they are not obvious. Prefer existing PyTorch, einops,
Hydra, and Transformers patterns over new abstractions. Keep public helpers
side-effect free where practical and avoid importing or initializing model
weights at module import time. Follow the repository's 100-character Ruff
line limit:

```bash
uv tool run ruff check src
uv tool run ruff check --select F --exclude exp_fourier.py tutorials tests
```

The current checkout does not contain a `.pre-commit-config.yaml`; do not
assume a `pre-commit` command is available unless that configuration is added.
Ruff is not in the locked development dependencies; `uv tool run` runs it in
an isolated tool environment. The tutorials retain compact teaching style
that does not yet pass all configured Ruff rules. The exclusion above leaves
the existing, untracked Fourier experiment outside this regression check.
`ruff check src` also reports pre-existing findings; do not mix unrelated
library lint fixes into a tutorial change.
Update the relevant README or tutorial when adding a new user-facing script,
inference mode, dependency, or known limitation.

## Changes, Tests, and Security

Keep changes focused and avoid committing commented-out experiments. For
inference changes, record the clip, `t0_us`, sampling parameters, hardware,
and observed metric or output. For visual or trajectory changes, include a
before/after plot or trajectory comparison when practical. Do not commit
Hugging Face tokens, model weights, dataset/cache files, generated videos, or
other outputs. Report vulnerabilities through `SECURITY.md`, not a public
issue.

Use concise imperative commit titles, include the corresponding issue number
when applicable, and sign off commits with `git commit -s`. Pull requests
should describe the behavior change, validation performed, and affected
standard, navigation, or VQA workflow. Follow the issue and review process in
`CONTRIBUTING.md`.
