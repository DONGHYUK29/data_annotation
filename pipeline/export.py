"""정제된 output_2 를 dataset/ 으로 모으기 (배경 프리픽스)."""
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


def transfer(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export stage2 → merged dataset")
    parser.add_argument(
        "--bg",
        type=str,
        required=True,
        help="배경 접두사 (예: paper, floor, desk)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "move"],
    )
    parser.add_argument("--src", type=Path, default=None, help="stage2 root (default cfg.STAGE2_DIR)")
    parser.add_argument("--dst", type=Path, default=None, help="dataset root (default cfg.DATASET_DIR)")
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
        new_name = f"{args.bg}_{clean_name}"
        stem = Path(clean_name).stem

        image_src = src_image / clean_name
        label_name = f"{stem}.txt"
        label_src = src_label / label_name

        image_dst = dst_image / new_name
        mask_dst = dst_mask / new_name
        label_dst = dst_label / new_name.replace(".png", ".txt")

        if image_src.exists():
            transfer(image_src, image_dst, args.mode)
        else:
            print("Missing image:", image_src)

        transfer(mask_path, mask_dst, args.mode)

        if label_src.exists():
            transfer(label_src, label_dst, args.mode)
        else:
            print("Missing label:", label_src)

        count += 1

    print("\nExported:", count)
    print("Mode:", args.mode)
    print("Dataset:", dst_root)


if __name__ == "__main__":
    main()
