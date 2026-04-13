"""(.bag) → RGB 이미지 + RGB-D 원본 추출."""
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


def process_bag(
    bag_path: Path,
    rgb_output_dir: Path,
    rgbd_output_dir: Path,
    target_count: int,
) -> None:
    class_id = bag_path.stem

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(str(bag_path), repeat_playback=False)

    rs_config.enable_stream(rs.stream.color)
    rs_config.enable_stream(rs.stream.depth)

    profile = pipeline.start(rs_config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align_to_color = rs.align(rs.stream.color) # rgb 기준으로 align

    frames_all: list[tuple[np.ndarray, np.ndarray]] = []

    print("\nReading bag:", bag_path)

    while True:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            break

        try:
            aligned_frames = align_to_color.process(frames)
        except RuntimeError:
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        frames_all.append((color_image.copy(), depth_image.copy()))

    pipeline.stop()

    total_frames = len(frames_all)
    print("Total aligned RGB-D frames:", total_frames)

    if total_frames == 0:
        print("No RGB-D frames found.")
        return

    if target_count >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = list(np.linspace(0, total_frames - 1, target_count, dtype=int))

    rgb_output_dir.mkdir(parents=True, exist_ok=True)
    rgbd_output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame_idx in enumerate(indices, start=1):
        color_bgr, depth_aligned = frames_all[frame_idx]

        stem = f"{class_id}_{i}"

        # RGB png 저장
        rgb_save_path = rgb_output_dir / f"{stem}.png"
        cv2.imwrite(str(rgb_save_path), color_bgr)
        # RGB-D 원본 저장 (.npz)
        rgbd_save_path = rgbd_output_dir / f"{stem}.npz"
        np.savez_compressed(
            rgbd_save_path,
            color=color_bgr,
            depth=depth_aligned,
        )

    print("Saved RGB:", len(indices), "->", rgb_output_dir)
    print("Saved RGB-D:", len(indices), "->", rgbd_output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract RGB and RGB-D frames from RealSense .bag files")
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
        "--rgb-out",
        type=Path,
        default=None,
        help=f"RGB output directory (default: {cfg.INPUT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--rgbd-out",
        type=Path,
        default=None,
        help=f"RGB-D output directory (default: {cfg.INPUT_RGBD_DIR})",
    )
    args = parser.parse_args(argv)

    bag_dir = args.bag_dir or cfg.BAG_DIR
    rgb_out_dir = args.rgb_out or cfg.INPUT_IMAGES_DIR
    rgbd_out_dir = args.rgbd_out or cfg.INPUT_RGBD_DIR

    for i in range(args.start, args.end + 1):
        bag_path = bag_dir / f"{i}.bag"
        if not bag_path.exists():
            print("Skip (not found):", bag_path)
            continue

        process_bag(
            bag_path=bag_path,
            rgb_output_dir=rgb_out_dir,
            rgbd_output_dir=rgbd_out_dir,
            target_count=args.count,
        )


if __name__ == "__main__":
    main()