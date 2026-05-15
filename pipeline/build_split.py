"""dataset/ → training/ train·val 분할 + dataset.yaml (항상 복사, 라벨에서 bbox 접두 제거)."""
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
from pipeline.label_utils import normalize_label_text

random.seed(42)


def list_images(src_dir: Path) -> list[Path]:
    imgs: list[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        imgs += list(src_dir.glob(ext))
    return sorted(imgs)


def read_class_id_from_label(label_path: Path) -> int:
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            return int(parts[0])
    raise ValueError(f"Label has no class id: {label_path}")


def parse_group(stem: str, label_path: Path):
    parts = stem.split("_")
    background = parts[0] if len(parts) >= 3 else "default"
    class_id = read_class_id_from_label(label_path)
    return (background, class_id)


def write_label_normalized(src_label: Path, dst_label: Path) -> None:
    """YOLO-seg 학습용: bbox(xc,yc,w,h) 접두가 있으면 제거한 뒤 저장."""
    text = src_label.read_text(encoding="utf-8")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text(normalize_label_text(text), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/val split from dataset/ (copy only)")
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

    samples = []
    for img_path in img_files:
        label_path = src_label / (img_path.stem + ".txt")
        if label_path.exists():
            samples.append((img_path, label_path))

    print("Valid samples:", len(samples))

    groups: dict[tuple, list] = defaultdict(list)
    for img_path, label_path in samples:
        key = parse_group(img_path.stem, label_path)
        groups[key].append((img_path, label_path))

    train_set = []
    val_set = []

    for _key, items in groups.items():
        random.shuffle(items)
        val_count = max(1, int(len(items) * val_ratio)) if items else 0
        val_set.extend(items[:val_count])
        train_set.extend(items[val_count:])

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))

    for img_path, label_path in train_set:
        shutil.copy2(img_path, img_train / img_path.name)
        write_label_normalized(label_path, lab_train / label_path.name)

    for img_path, label_path in val_set:
        shutil.copy2(img_path, img_val / img_path.name)
        write_label_normalized(label_path, lab_val / label_path.name)

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
