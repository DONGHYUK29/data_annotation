"""클래스별 샘플 수 줄이기 (무작위 삭제)."""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def extract_class_id(stem: str):
    for p in stem.split("_"):
        if p.isdigit():
            return int(p)
    return None


def trim_stage_dir(stage_root: Path, keep_per_class: int, seed: int = 42) -> None:
    random.seed(seed)
    label_dir = stage_root / "labels"
    mask_dir = stage_root / "masks"
    overlay_dir = stage_root / "images"

    class_map: dict[int, list[str]] = defaultdict(list)
    for f in label_dir.glob("*.txt"):
        cls = extract_class_id(f.stem)
        if cls is None:
            continue
        class_map[cls].append(f.stem)

    total_deleted = 0
    for cls, stems in class_map.items():
        if len(stems) <= keep_per_class:
            print(f"class {cls}: {len(stems)} (skip)")
            continue
        random.shuffle(stems)
        delete = stems[keep_per_class:]
        print(f"class {cls}: {len(stems)} → keep {keep_per_class}, delete {len(delete)}")
        for stem in delete:
            for folder in (label_dir, mask_dir, overlay_dir):
                ext = ".txt" if folder == label_dir else ".png"
                path = folder / f"{stem}{ext}"
                if path.exists():
                    path.unlink()
            total_deleted += 1
    print("\nDeleted:", total_deleted)


def trim_dataset_stack(dataset_root: Path | None = None, seed: int = 42) -> None:
    """config.STACK_TRIM_KEEP_PER_BG 기준으로 배경×클래스 그룹에서 초과분 삭제."""
    random.seed(seed)

    dataset_root = dataset_root or cfg.DATASET_DIR
    img_dir = dataset_root / "images"
    lab_dir = dataset_root / "labels"
    msk_dir = dataset_root / "masks"

    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for img in img_dir.glob("*.png"):
        parts = img.stem.split("_")
        if len(parts) < 3:
            continue
        bg = parts[0]
        cls = int(parts[1])
        groups[(bg, cls)].append(img.stem)

    total_deleted = 0
    for (bg, cls), stems in groups.items():
        keep_n = cfg.STACK_TRIM_KEEP_PER_BG.get(bg)
        if keep_n is None:
            continue
        if len(stems) <= keep_n:
            print(f"{bg} class{cls}: {len(stems)} (skip)")
            continue
        random.shuffle(stems)
        delete = stems[keep_n:]
        print(f"{bg} class{cls}: {len(stems)} → keep {keep_n}, delete {len(delete)}")
        for stem in delete:
            for path in (
                img_dir / f"{stem}.png",
                lab_dir / f"{stem}.txt",
                msk_dir / f"{stem}.png",
            ):
                if path.exists():
                    path.unlink()
            total_deleted += 1
    print("\nDeleted:", total_deleted)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trim samples")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stage = sub.add_parser("stage", help="Trim output_1 or output_2 style folder")
    p_stage.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Stage root containing labels/, masks/, images/",
    )
    p_stage.add_argument("--keep", type=int, default=200)
    p_stage.add_argument("--seed", type=int, default=42)

    p_ds = sub.add_parser(
        "dataset",
        help="Trim merged dataset using config.STACK_TRIM_KEEP_PER_BG",
    )
    p_ds.add_argument("--dir", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "stage":
        trim_stage_dir(args.dir, args.keep, args.seed)
    else:
        trim_dataset_stack(args.dir)


if __name__ == "__main__":
    main()
