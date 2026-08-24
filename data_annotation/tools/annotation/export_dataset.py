import os
import shutil
from pathlib import Path
import argparse


# =========================
# Paths
# =========================

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(
    os.environ.get("KERI_ANNOTATION_WORKDIR", REPO_ROOT / "data" / "annotation_work")
).expanduser().resolve()

SRC_IMAGE_DIR = BASE_DIR / "output_2" / "images"
SRC_MASK_DIR  = BASE_DIR / "output_2" / "masks"
SRC_LABEL_DIR = BASE_DIR / "output_2" / "labels"

DATASET_DIR = BASE_DIR / "dataset"

DST_IMAGE_DIR = DATASET_DIR / "images"
DST_MASK_DIR  = DATASET_DIR / "masks"
DST_LABEL_DIR = DATASET_DIR / "labels"

DST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DST_MASK_DIR.mkdir(parents=True, exist_ok=True)
DST_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CLI
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    default="copy",
    choices=["copy", "move"]
)

parser.add_argument(
    "--bg",
    type=str,
    required=True,
    help="background prefix (paper / floor / desk)"
)

args = parser.parse_args()

MODE = args.mode
BG = args.bg


# =========================
# Utils
# =========================

def clean_filename(name):
    """
    1_15_edited.png -> 1_15.png
    """
    if name.endswith("_edited.png"):
        return name.replace("_edited.png", ".png")
    return name


def transfer(src, dst):

    if MODE == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


# =========================
# Main
# =========================

def main():

    mask_files = sorted(SRC_MASK_DIR.glob("*.png"))

    print("Total masks:", len(mask_files))
    print("Background prefix:", BG)

    count = 0

    for mask_path in mask_files:

        # 1_15_edited.png -> 1_15.png
        clean_name = clean_filename(mask_path.name)

        # prefix 붙이기
        new_name = f"{BG}_{clean_name}"

        stem = clean_name.replace(".png", "")

        image_src = SRC_IMAGE_DIR / clean_name
        mask_src  = mask_path

        label_name = f"{stem}.txt"
        label_src  = SRC_LABEL_DIR / label_name

        image_dst = DST_IMAGE_DIR / new_name
        mask_dst  = DST_MASK_DIR / new_name
        label_dst = DST_LABEL_DIR / new_name.replace(".png", ".txt")

        # image copy
        if image_src.exists():
            transfer(image_src, image_dst)
        else:
            print("Missing image:", image_src)

        # mask copy
        transfer(mask_src, mask_dst)

        # label copy
        if label_src.exists():
            transfer(label_src, label_dst)
        else:
            print("Missing label:", label_src)

        count += 1


    print("\nExported:", count)
    print("Mode:", MODE)
    print("Dataset:", DATASET_DIR)


if __name__ == "__main__":
    main()
