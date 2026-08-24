#!/usr/bin/env python3
"""Create an Ultralytics-compatible depth-as-image dataset.

Input layout:
  source/
    depth/*.depth  # numpy uint16 arrays
    labels/*.txt

Output layout:
  output/
    images/*.png  # 8-bit 3-channel grayscale depth visualization
    labels/*.txt
    dataset.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def normalize_depth_to_u8(depth: np.ndarray, min_m: float, max_m: float, scale: float) -> np.ndarray:
    depth_m = depth.astype(np.float32) * scale
    valid = np.isfinite(depth_m) & (depth_m > min_m) & (depth_m < max_m)
    out = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    if int(valid.sum()) == 0:
        return out
    clipped = np.clip(depth_m, min_m, max_m)
    # Near objects become bright, far background becomes dark.
    norm = (max_m - clipped) / max(max_m - min_m, 1e-6)
    out[valid] = np.round(norm[valid] * 255.0).astype(np.uint8)
    return out


def write_dataset_yaml(output_dir: Path) -> None:
    text = """path: {path}

train: images
val: images
test: images

nc: 10
names:
  0: class_0
  1: class_1
  2: class_2
  3: class_3
  4: class_4
  5: class_5
  6: class_6
  7: class_7
  8: class_8
  9: class_9
""".format(path=output_dir)
    (output_dir / "dataset.yaml").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-m", type=float, default=0.10)
    parser.add_argument("--max-m", type=float, default=3.00)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    depth_dir = source / "depth"
    label_dir = source / "labels"
    out_image_dir = output / "images"
    out_label_dir = output / "labels"

    if not depth_dir.is_dir():
        raise RuntimeError(f"Missing depth directory: {depth_dir}")
    if not label_dir.is_dir():
        raise RuntimeError(f"Missing labels directory: {label_dir}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    depth_paths = sorted(depth_dir.glob("*.depth"))
    if not depth_paths:
        raise RuntimeError(f"No .depth files found in {depth_dir}")

    for i, depth_path in enumerate(depth_paths, 1):
        depth = np.load(depth_path)
        gray = normalize_depth_to_u8(depth, args.min_m, args.max_m, args.depth_scale)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out_image_path = out_image_dir / f"{depth_path.stem}.png"
        if not cv2.imwrite(str(out_image_path), rgb):
            raise RuntimeError(f"Failed to write {out_image_path}")

        src_label = label_dir / f"{depth_path.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, out_label_dir / src_label.name)
        else:
            (out_label_dir / f"{depth_path.stem}.txt").write_text("", encoding="utf-8")

        if i % 100 == 0 or i == len(depth_paths):
            print(f"Converted {i}/{len(depth_paths)}")

    write_dataset_yaml(output)
    print(f"Saved depth YOLO dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
