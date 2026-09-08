# Repository Guidelines

## Project Structure & Module Organization

Source code lives in `src/alpamayo1_5/`. Keep model orchestration in
`models/`, trajectory representations and kinematics in `action_space/`,
sampling code in `diffusion/`, and geometry helpers in `geometry/`.
`load_physical_aiavdataset.py` adapts the gated PhysicalAI AV dataset for
inference; `helper.py` builds Qwen chat inputs. Notebooks under `notebooks/`
demonstrate standard, navigation, camera-count, and VQA workflows. The
end-to-end example is `src/alpamayo1_5/test_inference.py`.

## Build, Test, and Development Commands

Use Python 3.12 and `uv`:

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
uv run ruff check src
python src/alpamayo1_5/test_inference.py
```

`uv sync` installs the locked runtime and development dependencies. The last
command is an inference example, not a lightweight unit test: it needs CUDA,
Hugging Face authentication, gated dataset access, and downloads about 22 GB
of model weights. Use `attn_implementation="sdpa"` when Flash Attention 2
cannot be built.

For this machine's operational quirks (China mirror endpoints, single-24GB-GPU
VRAM limits, headless notebook execution), see `RUN_NOTES.md`.

## Coding Style & Naming Conventions

Write Python with four-space indentation, type annotations, `snake_case` for
functions and variables, and `PascalCase` for classes. Keep tensors' expected
shapes in docstrings or short comments when they are non-obvious. Prefer
PyTorch and `einops` operations already used by neighboring modules over new
abstractions. Run `uv run ruff check src`; the repository config sets a
100-character line limit. Run `pre-commit format` before submitting changes,
as described in `CONTRIBUTING.md`.

## Testing Guidelines

There is currently no automated unit-test suite or CI configuration. Add
focused `pytest` tests in `tests/test_<module>.py` for pure geometry, token,
action-space, or diffusion behavior; avoid making normal tests depend on model
downloads, a GPU, or gated data. For inference changes, document the sampled
clip, parameters, hardware, and observed output or metric.

## Commit & Pull Request Guidelines

Recent history uses concise imperative summaries, sometimes with a scoped
prefix such as `docs(support): ...`, and may reference an issue or PR, for
example `#113 - Stabilize inferred curvature labels (#32)`. Keep commits
small and focused. Pull requests should state the behavioral change, validation
performed, affected inference mode (standard, navigation, or VQA), and linked
issue. Include plots or trajectory comparisons when a user-visible prediction
change is relevant.

## Security & Configuration

Do not commit Hugging Face tokens, downloaded weights, datasets, or generated
videos. Report vulnerabilities through the process in `SECURITY.md`, not a
public issue.
