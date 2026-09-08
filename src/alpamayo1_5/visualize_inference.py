# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an input, reasoning, and trajectory dashboard for Alpamayo inference."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.viz_utils import CAMERA_GRID_LAYOUT, make_camera_grid

BACKGROUND = "#141717"
PANEL = "#202626"
GRID = "#475050"
TEXT = "#F2F5F4"
MUTED = "#AAB5B3"
PREDICTION = "#FF6B4A"
GROUND_TRUTH = "#53D3C6"
HISTORY = "#8796A5"

CAMERA_LABELS = {
    0: "CROSS LEFT",
    1: "FRONT WIDE",
    2: "CROSS RIGHT",
    6: "FRONT TELE",
}


@dataclass
class Dashboard:
    """A dashboard figure with artists that reveal trajectories over time."""

    figure: Figure
    prediction_lines: list[Line2D]
    ground_truth_line: Line2D
    prediction_marker: Line2D
    ground_truth_marker: Line2D
    progress_label: plt.Text
    prediction_xy: np.ndarray
    ground_truth_xy: np.ndarray
    camera_image: AxesImage
    camera_frames: torch.Tensor
    camera_indices: torch.Tensor
    relative_timestamps: torch.Tensor
    camera_caption: plt.Text

    def set_progress(self, n_points: int) -> None:
        """Reveal a prefix of the 6.4-second future horizon."""
        n_points = max(0, min(n_points, self.ground_truth_xy.shape[0]))
        for line, trajectory in zip(self.prediction_lines, self.prediction_xy, strict=True):
            line.set_data(trajectory[:n_points, 0], trajectory[:n_points, 1])
        self.ground_truth_line.set_data(
            self.ground_truth_xy[:n_points, 0], self.ground_truth_xy[:n_points, 1]
        )

        if n_points:
            prediction = np.median(self.prediction_xy[:, n_points - 1], axis=0)
            ground_truth = self.ground_truth_xy[n_points - 1]
            self.prediction_marker.set_data([prediction[0]], [prediction[1]])
            self.ground_truth_marker.set_data([ground_truth[0]], [ground_truth[1]])
        else:
            self.prediction_marker.set_data([], [])
            self.ground_truth_marker.set_data([], [])

        self.progress_label.set_text(f"PREDICTION HORIZON  {n_points / 10:.1f} / 6.4 s")

    def set_camera_frame(self, frame_idx: int) -> None:
        """Show the ``frame_idx``-th temporal frame across all cameras."""
        num_frames = self.camera_frames.shape[1]
        frame_idx = max(0, min(frame_idx, num_frames - 1))
        grid = make_camera_grid(self.camera_frames, self.camera_indices, frame_idx=frame_idx)
        self.camera_image.set_data(grid)
        # Time offset of this frame relative to t0 (the latest frame).
        rel = float(self.relative_timestamps[0, frame_idx]) - float(
            self.relative_timestamps[0, -1]
        )
        label = "t0" if abs(rel) < 1e-6 else f"t0{rel:+.1f}s"
        self.camera_caption.set_text(f"4 cameras x {num_frames} temporal frames  |  {label}")

    def rgb_frame(self) -> np.ndarray:
        """Return the current dashboard raster as an RGB frame."""
        self.figure.canvas.draw()
        rgba = np.asarray(self.figure.canvas.buffer_rgba())
        return rgba[..., :3].copy()


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _camera_labels(ax: plt.Axes, image_frames: torch.Tensor, camera_indices: torch.Tensor) -> None:
    """Overlay semantic labels on the camera grid."""
    _, _, _, height, width = image_frames.shape
    for camera_id, (row, col) in CAMERA_GRID_LAYOUT.items():
        if camera_id not in camera_indices.tolist():
            continue
        x = (col + 0.05) * width
        y = (row + 0.12) * height
        ax.text(
            x,
            y,
            CAMERA_LABELS[camera_id],
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            va="top",
            bbox={"facecolor": "#101313", "alpha": 0.78, "edgecolor": "none", "pad": 3},
        )


def _trajectory_limits(*trajectories: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate([trajectory.reshape(-1, 2) for trajectory in trajectories], axis=0)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    half_extent = max((xmax - xmin) / 2, (ymax - ymin) / 2, 5.0) + 1.5
    return (x_center - half_extent, x_center + half_extent), (y_center - half_extent, y_center + half_extent)


def _extract_xy(data: dict, pred_xyz: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = data["ego_history_xyz"].cpu().numpy()[0, 0, :, :2]
    ground_truth = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]
    prediction = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    return history, ground_truth, prediction


def _extract_cot(extra: dict) -> str:
    cot = np.asarray(extra.get("cot", [[""]])).reshape(-1)
    return str(cot[0]) if cot.size else "No chain-of-causation text returned."


def build_dashboard(data: dict, pred_xyz: torch.Tensor, extra: dict, clip_id: str) -> Dashboard:
    """Build the static dashboard and return its animated trajectory artists."""
    history, ground_truth, prediction = _extract_xy(data, pred_xyz)
    min_ade = np.linalg.norm(prediction - ground_truth[None, :, :], axis=-1).mean(axis=1).min()
    camera_grid = make_camera_grid(data["image_frames"], data["camera_indices"])

    figure = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BACKGROUND)
    layout = figure.add_gridspec(
        24, 32, left=0.025, right=0.985, top=0.95, bottom=0.045, wspace=0.65, hspace=1.25
    )
    header = figure.add_subplot(layout[0:2, :])
    camera_ax = figure.add_subplot(layout[3:15, 0:20])
    reasoning_ax = figure.add_subplot(layout[3:15, 21:32])
    bev_ax = figure.add_subplot(layout[16:24, 0:21])
    output_ax = figure.add_subplot(layout[16:24, 22:32])

    header.set_facecolor(BACKGROUND)
    header.axis("off")
    header.text(
        0.0,
        0.78,
        "ALPAMAYO 1.5  /  AUTONOMOUS DRIVING INFERENCE",
        color=TEXT,
        fontsize=22,
        fontweight="bold",
        transform=header.transAxes,
    )
    header.text(
        0.0,
        0.20,
        f"CLIP {clip_id}   |   LOCAL EGO FRAME   |   SDPA BACKEND",
        color=MUTED,
        fontsize=10,
        transform=header.transAxes,
    )
    header.text(
        1.0,
        0.78,
        "INPUT  ->  REASONING  ->  TRAJECTORY",
        color=GROUND_TRUTH,
        fontsize=10,
        ha="right",
        transform=header.transAxes,
    )

    camera_ax.set_facecolor("#080909")
    camera_image = camera_ax.imshow(camera_grid)
    _camera_labels(camera_ax, data["image_frames"], data["camera_indices"])
    camera_ax.set_title("MULTI-CAMERA INPUT  /  t0", color=TEXT, fontsize=11, loc="left", pad=10)
    num_camera_frames = data["image_frames"].shape[1]
    camera_caption = camera_ax.text(
        0.995,
        0.02,
        f"4 cameras x {num_camera_frames} temporal frames  |  t0",
        color=MUTED,
        fontsize=8,
        ha="right",
        va="bottom",
        transform=camera_ax.transAxes,
        bbox={"facecolor": "#101313", "alpha": 0.8, "edgecolor": "none", "pad": 3},
    )
    camera_ax.axis("off")

    _style_axes(reasoning_ax)
    reasoning_ax.set_xticks([])
    reasoning_ax.set_yticks([])
    reasoning_ax.text(
        0.06,
        0.94,
        "CHAIN-OF-CAUSATION",
        color=GROUND_TRUTH,
        fontsize=11,
        fontweight="bold",
        va="top",
        transform=reasoning_ax.transAxes,
    )
    reasoning_ax.text(
        0.06,
        0.76,
        "\n".join(textwrap.wrap(_extract_cot(extra), width=31)),
        color=TEXT,
        fontsize=14,
        va="top",
        linespacing=1.5,
        transform=reasoning_ax.transAxes,
    )
    reasoning_ax.axhline(0.42, xmin=0.06, xmax=0.94, color=GRID, linewidth=1)
    reasoning_ax.text(
        0.06,
        0.34,
        "CONTEXT",
        color=MUTED,
        fontsize=9,
        fontweight="bold",
        transform=reasoning_ax.transAxes,
    )
    reasoning_ax.text(
        0.06,
        0.25,
        "1.6 s history\n16 ego states\n4 camera streams",
        color=TEXT,
        fontsize=11,
        va="top",
        linespacing=1.45,
        transform=reasoning_ax.transAxes,
    )
    reasoning_ax.text(
        0.55,
        0.34,
        "OUTPUT",
        color=MUTED,
        fontsize=9,
        fontweight="bold",
        transform=reasoning_ax.transAxes,
    )
    reasoning_ax.text(
        0.55,
        0.25,
        "6.4 s horizon\n64 waypoints\n1 trajectory sample",
        color=TEXT,
        fontsize=11,
        va="top",
        linespacing=1.45,
        transform=reasoning_ax.transAxes,
    )

    _style_axes(bev_ax)
    bev_ax.set_title("EGO-CENTRIC BIRD'S-EYE VIEW", color=TEXT, fontsize=11, loc="left", pad=10)
    bev_ax.set_xlabel("FORWARD X (m)", color=MUTED, fontsize=9)
    bev_ax.set_ylabel("LATERAL Y (m)", color=MUTED, fontsize=9)
    bev_ax.grid(color=GRID, linewidth=0.6, alpha=0.55)
    bev_ax.set_aspect("equal")
    xlim, ylim = _trajectory_limits(history, ground_truth, prediction)
    bev_ax.set_xlim(*xlim)
    bev_ax.set_ylim(*ylim)
    bev_ax.plot(history[:, 0], history[:, 1], color=HISTORY, linewidth=2.0, linestyle="--")
    bev_ax.scatter(history[-1, 0], history[-1, 1], color=TEXT, s=36, marker="o", zorder=6)
    bev_ax.annotate(
        "EGO / t0",
        xy=(history[-1, 0], history[-1, 1]),
        xytext=(8, 8),
        textcoords="offset points",
        color=TEXT,
        fontsize=8,
    )
    prediction_lines = [
        bev_ax.plot([], [], color=PREDICTION, linewidth=2.4, alpha=0.9)[0]
        for _ in range(prediction.shape[0])
    ]
    ground_truth_line = bev_ax.plot([], [], color=GROUND_TRUTH, linewidth=2.4)[0]
    prediction_marker = bev_ax.plot([], [], color=PREDICTION, marker="o", markersize=5)[0]
    ground_truth_marker = bev_ax.plot([], [], color=GROUND_TRUTH, marker="o", markersize=5)[0]
    legend = bev_ax.legend(
        handles=[
            Line2D([0], [0], color=HISTORY, linewidth=2, linestyle="--", label="History"),
            Line2D([0], [0], color=PREDICTION, linewidth=2.4, label="Prediction"),
            Line2D([0], [0], color=GROUND_TRUTH, linewidth=2.4, label="Ground truth"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=8,
    )
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(TEXT)
    progress_label = bev_ax.text(
        0.99,
        0.03,
        "PREDICTION HORIZON  0.0 / 6.4 s",
        color=MUTED,
        fontsize=8,
        ha="right",
        transform=bev_ax.transAxes,
    )

    _style_axes(output_ax)
    output_ax.set_xticks([])
    output_ax.set_yticks([])
    output_ax.text(
        0.08,
        0.92,
        "TRAJECTORY QUALITY",
        color=GROUND_TRUTH,
        fontsize=11,
        fontweight="bold",
        va="top",
        transform=output_ax.transAxes,
    )
    output_ax.text(
        0.08,
        0.68,
        f"{min_ade:.2f} m",
        color=TEXT,
        fontsize=30,
        fontweight="bold",
        transform=output_ax.transAxes,
    )
    output_ax.text(
        0.08,
        0.57,
        "MINIMUM AVERAGE DISPLACEMENT ERROR",
        color=MUTED,
        fontsize=8,
        transform=output_ax.transAxes,
    )
    output_ax.axhline(0.45, xmin=0.08, xmax=0.92, color=GRID, linewidth=1)
    output_ax.text(
        0.08,
        0.33,
        "PREDICTION",
        color=MUTED,
        fontsize=8,
        fontweight="bold",
        transform=output_ax.transAxes,
    )
    output_ax.text(
        0.08,
        0.23,
        "ACCELERATION + CURVATURE\n-> KINEMATIC TRAJECTORY",
        color=TEXT,
        fontsize=10,
        linespacing=1.45,
        transform=output_ax.transAxes,
    )

    dashboard = Dashboard(
        figure=figure,
        prediction_lines=prediction_lines,
        ground_truth_line=ground_truth_line,
        prediction_marker=prediction_marker,
        ground_truth_marker=ground_truth_marker,
        progress_label=progress_label,
        prediction_xy=prediction,
        ground_truth_xy=ground_truth,
        camera_image=camera_image,
        camera_frames=data["image_frames"],
        camera_indices=data["camera_indices"],
        relative_timestamps=data["relative_timestamps"],
        camera_caption=camera_caption,
    )
    dashboard.set_progress(0)
    return dashboard


def _video_frames(dashboard: Dashboard) -> Iterator[np.ndarray]:
    """Yield a trajectory-reveal animation with an input-camera playback intro."""
    num_camera_frames = dashboard.camera_frames.shape[1]
    for frame_idx in range(num_camera_frames):
        dashboard.set_camera_frame(frame_idx)
        dashboard.set_progress(0)
        for _ in range(3):
            yield dashboard.rgb_frame()
    for n_points in range(1, dashboard.ground_truth_xy.shape[0] + 1):
        dashboard.set_progress(n_points)
        yield dashboard.rgb_frame()
    for _ in range(24):
        dashboard.set_progress(dashboard.ground_truth_xy.shape[0])
        yield dashboard.rgb_frame()


def run_inference(clip_id: str, t0_us: int) -> tuple[dict, torch.Tensor, dict]:
    """Load the example sample and run one SDPA trajectory prediction."""
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    print("Dataset loaded.")
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
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
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    return data, pred_xyz, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/alpamayo1_5_inference")
    )
    parser.add_argument("--clip-id", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, pred_xyz, extra = run_inference(args.clip_id, args.t0_us)
    dashboard = build_dashboard(data, pred_xyz, extra, args.clip_id)

    dashboard.set_progress(dashboard.ground_truth_xy.shape[0])
    image_path = args.output_dir / "inference_dashboard.png"
    dashboard.figure.savefig(image_path, facecolor=BACKGROUND)
    print(f"Saved dashboard: {image_path}")

    video_path = args.output_dir / "inference_reveal.mp4"
    media.write_video(video_path, _video_frames(dashboard), fps=20, codec="h264", crf=20)
    print(f"Saved video: {video_path}")
    plt.close(dashboard.figure)


if __name__ == "__main__":
    main()
