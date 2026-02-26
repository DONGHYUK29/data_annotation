import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent  # .../data_annotation

ORIG_LABEL_DIR = BASE_DIR / "images" / "yolo_seg_results" / "labels"
EDIT_MASK_DIR = BASE_DIR / "images" / "edit_results" / "masks"
EDIT_LABEL_DIR = BASE_DIR / "images" / "edit_results" / "labels"


def load_original_class_ids():
    """
    Read YOLO seg result labels and build:
        instance_id -> class_id
    """
    mapping = {}

    if not ORIG_LABEL_DIR.exists():
        return mapping

    for txt_path in ORIG_LABEL_DIR.glob("*.txt"):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                inst_id = parts[0]
                try:
                    cls_id = int(float(parts[1]))
                except ValueError:
                    cls_id = 0
                mapping[inst_id] = cls_id

    return mapping


def parse_instance_id_from_mask_name(stem: str) -> str:
    """
    Examples:
      - "000001_obj0_mask_edit" -> "000001_obj0"
      - "000001_obj0_mask"      -> "000001_obj0"
    """
    name = stem
    if name.endswith("_edit"):
        name = name[: -len("_edit")]
    if name.endswith("_mask"):
        name = name[: -len("_mask")]
    return name


def compute_geometry(mask: np.ndarray):
    """
    Given a binary mask uint8 (0/255), compute:
      - bbox: x1, y1, x2, y2
      - polygon: list of (x, y) along the outer contour
    """
    if mask is None:
        return None, None

    # ensure binary 0/1
    bin_mask = (mask > 127).astype("uint8")

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # use largest contour
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    x1, y1, x2, y2 = x, y, x + w, y + h

    # squeeze to (N, 2)
    poly = contour.squeeze(1)
    if poly.ndim != 2 or poly.shape[1] != 2:
        return (x1, y1, x2, y2), None

    return (x1, y1, x2, y2), poly


def main():
    EDIT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    orig_cls_map = load_original_class_ids()
    if not EDIT_MASK_DIR.exists():
        print(f"No edited masks found in: {EDIT_MASK_DIR}")
        return

    # image_id -> list of label lines
    per_image_lines = defaultdict(list)

    for mask_path in sorted(EDIT_MASK_DIR.glob("*.png")):
        stem = mask_path.stem
        instance_id = parse_instance_id_from_mask_name(stem)

        # image_id: 앞부분 (예: "000001_obj0" -> "000001")
        if "_obj" in instance_id:
            image_id = instance_id.split("_obj")[0]
        else:
            image_id = instance_id

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Failed to read edited mask: {mask_path}")
            continue

        geom = compute_geometry(mask)
        if geom[0] is None:
            # empty mask
            continue

        bbox, poly = geom
        x1, y1, x2, y2 = bbox

        cls_id = orig_cls_map.get(instance_id, 0)

        if poly is not None:
            poly_flat = " ".join([f"{float(x):.2f} {float(y):.2f}" for x, y in poly])
        else:
            poly_flat = ""

        line = f"{instance_id} {cls_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
        if poly_flat:
            line += f" {poly_flat}"
        line += "\n"

        per_image_lines[image_id].append(line)

    # write per-image label files
    for image_id, lines in per_image_lines.items():
        out_path = EDIT_LABEL_DIR / f"{image_id}_edit.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                "# instance_id class_id x1 y1 x2 y2 "
                "poly_x0 poly_y0 poly_x1 poly_y1 ...\n"
            )
            for line in lines:
                f.write(line)

    print(f"Exported edited labels for {len(per_image_lines)} images into: {EDIT_LABEL_DIR}")


if __name__ == "__main__":
    main()


