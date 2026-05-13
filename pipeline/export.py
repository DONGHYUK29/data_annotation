"""정제된 output_2 를 dataset/ 으로 모으기 (배경 프리픽스, 항상 복사)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def clean_filename(name: str) -> str:
    if name.endswith("_edited.png"):
        return name.replace("_edited.png", ".png")
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export stage2 → merged dataset (copy only)")
    parser.add_argument(
        "--bg",
        type=str,
        default="",
        required=False,
        help="배경 접두사 (예: paper, floor, desk, Environment)",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="stage2 root (default cfg.STAGE2_DIR)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="dataset root (default cfg.DATASET_DIR)",
    )
    args = parser.parse_args(argv)

    src_root = args.src or cfg.STAGE2_DIR
    dst_root = args.dst or cfg.DATASET_DIR

    src_image = src_root / "images"
    src_mask = src_root / "masks"
    src_label = src_root / "labels"

    dst_image = dst_root / "images"
    dst_mask = dst_root / "masks"
    dst_label = dst_root / "labels"

    for p in (dst_image, dst_mask, dst_label):
        p.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(src_mask.glob("*.png"))
    print("Total masks:", len(mask_files))
    print("Background prefix:", args.bg)

    count = 0
    for mask_path in mask_files:
        clean_name = clean_filename(mask_path.name)
        stem = Path(clean_name).stem
        new_name = f"{args.bg}_{clean_name}"
        new_stem = Path(new_name).stem

        image_src = src_image / clean_name
        label_src = src_label / f"{stem}.txt"

        image_dst = dst_image / new_name
        mask_dst = dst_mask / new_name
        label_dst = dst_label / f"{new_stem}.txt"

        if image_src.exists():
            shutil.copy2(image_src, image_dst)
        else:
            print("Missing image:", image_src)

        shutil.copy2(mask_path, mask_dst)

        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            print("Missing label:", label_src)

        count += 1

    print("\nExported:", count)
    print("Dataset:", dst_root)


if __name__ == "__main__":
    main()
