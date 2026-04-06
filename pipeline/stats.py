"""데이터셋 이미지 파일명에서 클래스별 개수 집계."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count images per class (from filename digits)")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"images folder (default: {cfg.DATASET_IMAGES})",
    )
    args = parser.parse_args(argv)

    image_dir = args.dir or cfg.DATASET_IMAGES
    counts: dict[int, int] = defaultdict(int)

    for img_path in image_dir.glob("*.png"):
        parts = img_path.stem.split("_")
        cls = None
        for p in parts:
            if p.isdigit():
                cls = int(p)
                break
        if cls is None:
            continue
        counts[cls] += 1

    total = 0
    print("\nClass counts\n")
    for cls in sorted(counts.keys()):
        n = counts[cls]
        print(f"class {cls}: {n}")
        total += n
    print("\nTotal:", total)


if __name__ == "__main__":
    main()
