# -*- coding: utf-8 -*-
"""
extract_frames_from_video.py

mp4 영상에서 일정 간격으로 프레임 추출.
"""

from pathlib import Path
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--prefix", type=str, default="lab_bg")
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    interval = max(1, int(round(src_fps / args.sample_fps)))

    print("=" * 70)
    print("Extract frames")
    print(f"video     : {video_path}")
    print(f"out_dir   : {out_dir}")
    print(f"src_fps   : {src_fps:.2f}")
    print(f"sample_fps: {args.sample_fps}")
    print(f"interval  : {interval}")
    print("=" * 70)

    frame_idx = 0
    save_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % interval == 0:
            out_path = out_dir / f"{args.prefix}_{save_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            save_idx += 1

        frame_idx += 1

    cap.release()

    print("=" * 70)
    print("DONE")
    print(f"read frames : {frame_idx}")
    print(f"saved frames: {save_idx}")
    print(f"out_dir     : {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
