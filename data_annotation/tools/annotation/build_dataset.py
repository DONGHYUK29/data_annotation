import shutil
import random
from pathlib import Path
import argparse
from collections import defaultdict

random.seed(42)

# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "data" / "dataset"

SRC_IMAGE_DIR = BASE_DIR / "images"
SRC_LABEL_DIR = BASE_DIR / "labels"

TRAIN_DIR = BASE_DIR / "training"

IMG_TRAIN = TRAIN_DIR / "images" / "train"
IMG_VAL   = TRAIN_DIR / "images" / "val"

LAB_TRAIN = TRAIN_DIR / "labels" / "train"
LAB_VAL   = TRAIN_DIR / "labels" / "val"

for p in [IMG_TRAIN, IMG_VAL, LAB_TRAIN, LAB_VAL]:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# CLI
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    default="copy",
    choices=["copy", "move"]
)

parser.add_argument(
    "--val-ratio",
    type=float,
    default=0.1
)

parser.add_argument(
    "--num-classes",
    type=int,
    required=True
)

args = parser.parse_args()

MODE = args.mode
VAL_RATIO = args.val_ratio
NUM_CLASSES = args.num_classes


# =========================
# Utils
# =========================

def transfer(src, dst):

    if MODE == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def list_images():

    imgs = []

    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        imgs += list(SRC_IMAGE_DIR.glob(ext))

    return sorted(imgs)


def parse_group(stem):

    parts = stem.split("_")

    # prefix 없는 경우
    if len(parts) == 2:
        background = "default"
        class_id = int(parts[0])

    # prefix 있는 경우
    elif len(parts) >= 3:
        background = parts[0]
        class_id = int(parts[1])

    else:
        raise ValueError(f"Invalid filename: {stem}")

    return (background, class_id)


# =========================
# Main
# =========================

def main():

    img_files = list_images()

    print("Total images:", len(img_files))

    if len(img_files) == 0:
        print("No dataset found.")
        return


    # =========================
    # label 존재하는 것만 사용
    # =========================

    samples = []

    for img_path in img_files:

        label_path = SRC_LABEL_DIR / (img_path.stem + ".txt")

        if label_path.exists():
            samples.append((img_path, label_path))


    print("Valid samples:", len(samples))


    # =========================
    # stratified grouping
    # =========================

    groups = defaultdict(list)

    for img_path, label_path in samples:

        key = parse_group(img_path.stem)

        groups[key].append((img_path, label_path))


    train_set = []
    val_set = []


    for key, items in groups.items():

        random.shuffle(items)

        val_count = max(1, int(len(items) * VAL_RATIO))

        val_set.extend(items[:val_count])
        train_set.extend(items[val_count:])


    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))


    # =========================
    # copy / move
    # =========================

    for img_path, label_path in train_set:

        img_dst = IMG_TRAIN / img_path.name
        lab_dst = LAB_TRAIN / label_path.name

        transfer(img_path, img_dst)
        transfer(label_path, lab_dst)


    for img_path, label_path in val_set:

        img_dst = IMG_VAL / img_path.name
        lab_dst = LAB_VAL / label_path.name

        transfer(img_path, img_dst)
        transfer(label_path, lab_dst)


    # =========================
    # YAML 생성
    # =========================

    yaml_path = TRAIN_DIR / "dataset.yaml"

    names = [f"class_{i}" for i in range(NUM_CLASSES)]

    with open(yaml_path, "w") as f:

        f.write(f"path: {TRAIN_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")

        f.write(f"nc: {NUM_CLASSES}\n")

        f.write("names:\n")

        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")


    print("\nDataset build complete")
    print("Train:", len(train_set))
    print("Val:", len(val_set))
    print("Mode:", MODE)
    print("YAML:", yaml_path)


if __name__ == "__main__":
    main()
