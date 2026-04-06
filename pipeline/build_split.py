"""dataset/ → training/ train·val 분할 + dataset.yaml."""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

random.seed(42)


def transfer(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def list_images(src_dir: Path) -> list[Path]:
    imgs: list[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        imgs += list(src_dir.glob(ext))
    return sorted(imgs)


def parse_group(stem: str):
    parts = stem.split("_")
    background = parts[0]
    class_id = int(parts[1])
    return (background, class_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/val split from dataset/")
    parser.add_argument("--mode", default="copy", choices=["copy", "move"])
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help=f"default: {cfg.DEFAULT_VAL_RATIO}",
    )
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--dataset", type=Path, default=None, help="cfg.DATASET_DIR")
    parser.add_argument("--training", type=Path, default=None, help="cfg.TRAINING_DIR")
    args = parser.parse_args(argv)

    val_ratio = (
        args.val_ratio if args.val_ratio is not None else cfg.DEFAULT_VAL_RATIO
    )

    base = args.dataset or cfg.DATASET_DIR
    train_root = args.training or cfg.TRAINING_DIR

    src_image = base / "images"
    src_label = base / "labels"

    img_train = train_root / "images" / "train"
    img_val = train_root / "images" / "val"
    lab_train = train_root / "labels" / "train"
    lab_val = train_root / "labels" / "val"
    for p in (img_train, img_val, lab_train, lab_val):
        p.mkdir(parents=True, exist_ok=True)

    img_files = list_images(src_image)
    print("Total images:", len(img_files))
    if not img_files:
        print("No dataset found.")
        return

    samples = []
    for img_path in img_files:
        label_path = src_label / (img_path.stem + ".txt")
        if label_path.exists():
            samples.append((img_path, label_path))

    print("Valid samples:", len(samples))

    groups: dict[tuple, list] = defaultdict(list)
    for img_path, label_path in samples:
        key = parse_group(img_path.stem)
        groups[key].append((img_path, label_path))

    train_set = []
    val_set = []

    for _key, items in groups.items():
        random.shuffle(items)
        val_count = max(1, int(len(items) * val_ratio))
        val_set.extend(items[:val_count])
        train_set.extend(items[val_count:])

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    for img_path, label_path in train_set:
        transfer(img_path, img_train / img_path.name, args.mode)
        transfer(label_path, lab_train / label_path.name, args.mode)

    for img_path, label_path in val_set:
        transfer(img_path, img_val / img_path.name, args.mode)
        transfer(label_path, lab_val / label_path.name, args.mode)

    yaml_path = train_root / "dataset.yaml"
    names = [f"class_{i}" for i in range(args.num_classes)]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {train_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {args.num_classes}\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    print("\nDataset build complete")
    print("YAML:", yaml_path)


if __name__ == "__main__":
    main()
