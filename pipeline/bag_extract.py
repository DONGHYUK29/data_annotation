"""ROS bag (.bag) → 프레임 이미지 추출."""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def process_bag(bag_path: Path, output_dir: Path, target_count: int) -> None:
    class_id = bag_path.stem

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(str(bag_path), repeat_playback=False)
    rs_config.enable_stream(rs.stream.color)

    pipeline.start(rs_config)

    playback = pipeline.get_active_profile().get_device().as_playback()
    playback.set_real_time(False)

    frames_all: list[np.ndarray] = []

    print("\nReading bag:", bag_path)

    while True:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            break

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frames_all.append(image)

    pipeline.stop()

    total_frames = len(frames_all)
    print("Total frames:", total_frames)

    if total_frames == 0:
        print("No frames found.")
        return

    if target_count >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = list(
            np.linspace(0, total_frames - 1, target_count, dtype=int)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame_idx in enumerate(indices, start=1):
        image = frames_all[frame_idx]
        filename = f"{class_id}_{i}.png"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), image)

    print("Saved:", len(indices))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract frames from RealSense .bag files")
    parser.add_argument("--start", type=int, required=True, help="bag index start (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="bag index end (inclusive)")
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="number of images to sample per bag (uniformly along timeline)",
    )
    parser.add_argument(
        "--bag-dir",
        type=Path,
        default=None,
        help=f"override cfg.BAG_DIR (default: {cfg.BAG_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output directory (default: {cfg.IMAGES_RAW_DIR})",
    )
    args = parser.parse_args(argv)

    bag_dir = args.bag_dir or cfg.BAG_DIR
    out_dir = args.out or cfg.IMAGES_RAW_DIR

    for i in range(args.start, args.end + 1):
        bag_path = bag_dir / f"{i}.bag"
        if not bag_path.exists():
            print("Skip (not found):", bag_path)
            continue
        process_bag(bag_path, out_dir, args.count)


if __name__ == "__main__":
    main()
