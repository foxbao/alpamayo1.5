"""Pre-download chunk feature files for one or more clips into the HF cache.

Uses the hf-mirror.com endpoint directly (bypassing any proxy) to avoid the
slow proxy timeout that breaks the streaming path in the inference examples.

Usage:
    python download_clip_data.py [CLIP_ID ...]
"""
import os
import sys

# Use the China HF mirror and bypass the local proxy for it.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(key, None)

import physical_ai_av

DEFAULT_CLIPS = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]


def main() -> None:
    clips = sys.argv[1:] or DEFAULT_CLIPS
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        confirm_download_threshold_gb=float("inf"),
    )
    features = [
        avdi.features.LABELS.EGOMOTION,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]
    chunks = sorted({int(avdi.get_clip_chunk(c)) for c in clips})
    print(f"Clips: {clips}", flush=True)
    print(f"Chunks to download: {chunks}", flush=True)
    avdi.download_clip_features(clips, features, max_workers=4)
    print("All downloads complete.", flush=True)


if __name__ == "__main__":
    main()
