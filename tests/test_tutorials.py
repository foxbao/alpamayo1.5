"""Regression tests for deterministic, CPU-only tutorial behavior."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tutorials")]

import common  # noqa: E402
import ddpm_1d  # noqa: E402
import real3_kv_cache  # noqa: E402
import real5_eval_metrics  # noqa: E402
import real6_ablation  # noqa: E402
import stage11_fusion  # noqa: E402
import stage12_two_cameras  # noqa: E402
import stage15_cfg  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_ddpm_reverse_step_uses_posterior_noise() -> None:
    x_t = torch.zeros(1, 1)
    eps_pred = torch.zeros_like(x_t)
    t = 10

    no_noise = ddpm_1d.reverse_step(x_t, eps_pred, t, noise=torch.zeros_like(x_t))
    unit_noise = ddpm_1d.reverse_step(x_t, eps_pred, t, noise=torch.ones_like(x_t))

    assert torch.equal(no_noise, torch.zeros_like(x_t))
    assert torch.allclose(unit_noise, ddpm_1d.posterior_var[t].sqrt().reshape(1, 1))


@torch.no_grad()
def test_cfg_comparison_reuses_fixed_initial_noise() -> None:
    torch.manual_seed(0)
    model = stage15_cfg.MiniVLA(3).eval()
    left_condition = model.cond_gen(torch.tensor([0]))
    null_condition = model.cond_gen(torch.tensor([2]))
    noise = torch.randn(1, stage15_cfg.N_WAYPOINTS, stage15_cfg.ACTION_DIM)

    first = model.sample_cfg(left_condition, null_condition, 1.5, batch_size=1, noise=noise)
    second = model.sample_cfg(left_condition, null_condition, 1.5, batch_size=1, noise=noise)

    assert torch.equal(first, second)


def test_ddpm_final_step_returns_clean_sample_without_adding_noise() -> None:
    x0 = torch.tensor([[1.0], [-1.0]])
    eps = torch.tensor([[0.2], [0.4]])
    at = ddpm_1d.alpha_bar[0]
    x_t = at.sqrt() * x0 + (1 - at).sqrt() * eps
    actual = ddpm_1d.reverse_step(x_t, eps, 0, noise=torch.full_like(x0, 100.0))
    torch.testing.assert_close(actual, x0)


@pytest.mark.parametrize("w", [0.0, 1.0, 2.0])
def test_cfg_formula_and_condition_batch_expansion(w: float) -> None:
    model = stage15_cfg.MiniVLA(3)
    model.step_fn = lambda x, t, condition: torch.ones_like(x) * condition[:, :1, :1]
    cond = torch.full((1, 1, common.HIDDEN), 3.0)
    uncond = torch.full_like(cond, 1.0)
    noise = torch.zeros(2, common.N_WAYPOINTS, common.ACTION_DIM)
    actual = model.sample_cfg(cond, uncond, w, batch_size=2, n_steps=2, noise=noise)
    torch.testing.assert_close(actual, torch.full_like(noise, 1 + 2 * w))
    assert torch.count_nonzero(noise) == 0


def test_cfg_dropped_conditions_keep_both_target_modes(monkeypatch) -> None:
    torch.manual_seed(0)
    model = stage15_cfg.MiniVLA(3)
    observed = {}
    original_step = model.step_fn
    original_loss = torch.nn.functional.mse_loss

    def capture_ids(module, args):
        observed["ids"] = args[0].clone()

    def capture_step(x, t, condition):
        observed["x0"] = x.clone()
        return original_step(x, t, condition)

    def capture_loss(pred, target):
        observed["target"] = target.clone()
        return original_loss(pred, target)

    # t=0 reveals x0, while the same zero draws drop every condition.
    monkeypatch.setattr(torch, "rand", lambda *shape: torch.zeros(*shape))
    monkeypatch.setattr(model, "step_fn", capture_step)
    monkeypatch.setattr(stage15_cfg.F, "mse_loss", capture_loss)
    handle = model.cond_gen.register_forward_pre_hook(capture_ids)
    try:
        stage15_cfg.train(model, torch.optim.Adam(model.parameters()), n_iters=1, batch=16)
    finally:
        handle.remove()

    x1 = observed["target"] + observed["x0"]
    assert torch.all(observed["ids"] == 2)
    assert (x1[:, :, 1] > 0).any() and (x1[:, :, 1] < 0).any()
    torch.testing.assert_close(x1[:, :, 1].abs(), torch.full((16, 64), stage15_cfg.KAPPA))


def test_pairwise_endpoint_distance_is_not_centroid_distance() -> None:
    pred_xy = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [3.0, 4.0]],
        ]
    )

    pairwise, endpoints = real5_eval_metrics.pairwise_trajectory_distances(pred_xy)

    assert pairwise.shape == (1, 2)
    assert np.allclose(pairwise, [[1.0, 5.0]])
    assert np.allclose(endpoints, [5.0])


def test_pairwise_distances_cover_every_unordered_pair() -> None:
    pred_xy = np.array([[[0.0, 0.0]], [[3.0, 0.0]], [[0.0, 4.0]]])
    pairwise, endpoints = real5_eval_metrics.pairwise_trajectory_distances(pred_xy)
    np.testing.assert_allclose(pairwise[:, 0], [3.0, 4.0, 5.0])
    np.testing.assert_allclose(endpoints, pairwise[:, 0])


def test_pairwise_distances_with_one_sample_have_no_pairs() -> None:
    pairwise, endpoints = real5_eval_metrics.pairwise_trajectory_distances(np.zeros((1, 2, 2)))
    assert pairwise.shape == (0, 2)
    assert endpoints.shape == (0,)


def test_pairwise_distances_reject_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="T > 0"):
        real5_eval_metrics.pairwise_trajectory_distances(np.zeros((2, 0, 2)))


def test_cosmos_reason_rejects_sequences_beyond_positional_capacity() -> None:
    model = common.CosmosReason(vocab_size=7, max_len=3)
    with pytest.raises(ValueError, match="max_len"):
        model(
            torch.zeros(1, 2, common.HIDDEN),
            torch.zeros(1, 2, dtype=torch.long),
            common.make_causal_mask(4),
        )


@torch.no_grad()
def test_fusion_preserves_history_order() -> None:
    torch.manual_seed(0)
    model = stage11_fusion.MiniVLA().eval()
    history = stage11_fusion.HISTORIES[:1]
    image = stage11_fusion.IMAGES[:1]
    text = stage11_fusion.INSTRUCTIONS[:1]
    actions = torch.randn(1, common.N_WAYPOINTS, common.HIDDEN)
    ordered = model.expert(actions, model.encode(history, image, text))
    reversed_history = model.expert(actions, model.encode(history.flip(1), image, text))
    assert not torch.allclose(ordered, reversed_history, atol=1e-7, rtol=1e-7)


@torch.no_grad()
def test_camera_labels_are_bound_to_their_image_tokens() -> None:
    torch.manual_seed(0)
    model = stage12_two_cameras.MiniVLA().eval()
    front = stage12_two_cameras.FRONT[:1]
    side = stage12_two_cameras.SIDE[:1]
    actions = torch.randn(1, common.N_WAYPOINTS, common.HIDDEN)
    condition = model.encode(front, side)
    swapped = model.encode(side, front)

    # Bare concat would make the swapped input a K/V permutation with identical output.
    assert not torch.allclose(
        model.expert(actions, condition), model.expert(actions, swapped), atol=1e-6, rtol=1e-6
    )
    # Even identical images receive different features after reading front/side labels.
    same_image = model.encode(front, front)
    assert not torch.allclose(same_image[:, 1:18], same_image[:, 19:36])


def test_kv_cache_matches_recomputation() -> None:
    torch.manual_seed(0)
    recompute = real3_kv_cache.CrossAttnRecompute(hidden=16)
    cached = real3_kv_cache.CrossAttnKVCache(hidden=16)
    cached.load_state_dict(recompute.state_dict())
    x = torch.randn(1, 4, 16)
    cond = torch.randn(1, 3, 16)
    expected = real3_kv_cache.run_denoise(recompute, x, cond, use_cache=False)
    actual = real3_kv_cache.run_denoise(cached, x, cond, use_cache=True)
    torch.testing.assert_close(actual, expected)
    assert recompute.kv_calls == real3_kv_cache.N_STEPS
    assert cached.kv_calls == 1
    assert not actual.requires_grad


def test_ablation_noise_is_independent_of_text_rng_consumption() -> None:
    torch.manual_seed(0)
    torch.rand(3)
    first = real6_ablation.sample_with_fixed_seed(torch.randn, 1, 64, 2)
    torch.rand(100)
    second = real6_ablation.sample_with_fixed_seed(torch.randn, 1, 64, 2)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "module_name",
    [
        "stage6_training", "stage7_history_condition", "stage8_multimodal", "stage9_vision",
        "stage10_text", "stage11_fusion", "stage12_two_cameras", "stage13_coc",
        "stage14_complete", "stage15_cfg",
    ],
)
def test_training_stage_one_step(module_name: str) -> None:
    module = importlib.import_module(module_name)
    model = module.MiniVLA(3) if module_name in {"stage6_training", "stage15_cfg"} else module.MiniVLA()
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if module_name in {"stage8_multimodal", "stage15_cfg"}:
        module.train(model, opt, n_iters=1, batch=2)
    else:
        target = torch.zeros(3, common.N_WAYPOINTS, common.ACTION_DIM)
        target[0, :, 1] = 0.1
        target[1, :, 1] = -0.1
        module.train(model, opt, target, n_iters=1, batch=2)
    assert model.training
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
