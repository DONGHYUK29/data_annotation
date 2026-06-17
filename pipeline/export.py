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
from pipeline.label_utils import normalize_label_text


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def clean_filename(name: str) -> str:
    if name.endswith("_edited.png"):
        return name.replace("_edited.png", ".png")
    return name


def list_images(src_dir: Path) -> list[Path]:
    files: dict[Path, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for path in src_dir.glob(f"*{ext}"):
            files[path.resolve()] = path
        for path in src_dir.glob(f"*{ext.upper()}"):
            files[path.resolve()] = path
    return sorted(files.values())


def find_mask(src_mask: Path, stem: str) -> Path | None:
    candidates = [
        src_mask / f"{stem}_edited.png",
        src_mask / f"{stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def prefixed_name(bg: str, name: str) -> str:
    bg = bg.strip()
    return f"{bg}_{name}" if bg else name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export output2 → merged dataset (copy only)")
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
        help="output2 root (default cfg.output2_DIR)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="dataset root (default cfg.DATASET_DIR)",
    )
    args = parser.parse_args(argv)

    src_root = args.src or cfg.OUTPUT2_DIR
    dst_root = args.dst or cfg.DATASET_DIR

    src_image = src_root / "images"
    src_mask = src_root / "masks"
    src_label = src_root / "labels"

    dst_image = dst_root / "images"
    dst_mask = dst_root / "masks"
    dst_label = dst_root / "labels"

    for p in (dst_image, dst_mask, dst_label):
        p.mkdir(parents=True, exist_ok=True)

    image_files = list_images(src_image)
    print("Total images:", len(image_files))
    print("Background prefix:", args.bg)

    count = 0
    for image_src in image_files:
        clean_name = clean_filename(image_src.name)
        stem = image_src.stem
        new_name = prefixed_name(args.bg, clean_name)
        new_stem = Path(new_name).stem

        label_src = src_label / f"{stem}.txt"
        mask_src = find_mask(src_mask, stem)

        image_dst = dst_image / new_name
        mask_dst = dst_mask / new_name
        label_dst = dst_label / f"{new_stem}.txt"

        if not label_src.exists():
            print("Missing label:", label_src)
            continue

        shutil.copy2(image_src, image_dst)

        if mask_src is not None:
            shutil.copy2(mask_src, mask_dst)

        label_dst.write_text(
            normalize_label_text(label_src.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

        count += 1

    print("\nExported:", count)
    print("Dataset:", dst_root)


if __name__ == "__main__":
    main()
