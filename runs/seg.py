# runs/seg.py
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent  # .../data_annotation
ORIGINAL_DIR = BASE_DIR / "images" / "original"
YOLO_SEG_LABEL_DIR = BASE_DIR / "images" / "yolo_seg_results" / "labels"
YOLO_SEG_MASK_DIR = BASE_DIR / "images" / "yolo_seg_results" / "masks"


def ensure_dirs() -> None:
    """Ensure output directories for YOLO seg results exist."""
    YOLO_SEG_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_SEG_MASK_DIR.mkdir(parents=True, exist_ok=True)


def save_instance(
    image_id: str,
    inst_idx: int,
    img_bgr: np.ndarray,
    result,
    label_f,
) -> None:
    """
    Save one instance:
    - binary mask PNG: images/yolo_seg_results/masks/{instance_id}_mask.png
    - one line of metadata into opened label file

    Line format (space separated):
        instance_id class_id x1 y1 x2 y2  poly_x0 poly_y0 poly_x1 poly_y1 ...
    """
    # instance / class / bbox
    instance_id = f"{image_id}_obj{inst_idx}"
    cls_id = int(result.boxes.cls[inst_idx].item())

    x1, y1, x2, y2 = result.boxes.xyxy[inst_idx].tolist()

    # polygon from ultralytics (already in image coords)
    polys = result.masks.xy[inst_idx]  # (N, 2)
    poly_flat = " ".join([f"{float(x):.2f} {float(y):.2f}" for x, y in polys])

    # binary mask
    mask_data = result.masks.data[inst_idx].cpu().numpy()
    mask_u8 = (mask_data > 0.5).astype("uint8") * 255

    # safety: resize to image size
    h, w = img_bgr.shape[:2]
    mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_name = f"{instance_id}_mask.png"
    cv2.imwrite(str(YOLO_SEG_MASK_DIR / mask_name), mask_u8)

    # write one line of metadata
    line = f"{instance_id} {cls_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {poly_flat}\n"
    label_f.write(line)


def main():
    ensure_dirs()

    model = YOLO(str(BASE_DIR / "weights" / "yolo26l-seg.pt"))

    img_paths = sorted(ORIGINAL_DIR.glob("*.jpg"))
    img_count = len(img_paths)
    assert img_count > 0, f"No input images found in: {ORIGINAL_DIR}"

    print(f"[YOLO-SEG] images: {img_count}")

    t0 = time.time()

    for img_path in img_paths:
        image_id = img_path.stem  # e.g. 000001
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        result = model(img, verbose=False, retina_masks=True)[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            # nothing detected -> skip creating label file
            continue

        label_path = YOLO_SEG_LABEL_DIR / f"{image_id}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            # optional header (commented, starts with '#')
            f.write(
                "# instance_id class_id x1 y1 x2 y2 "
                "poly_x0 poly_y0 poly_x1 poly_y1 ...\n"
            )

            num_instances = len(result.boxes)
            for i in range(num_instances):
                save_instance(image_id, i, img, result, f)

    t1 = time.time()
    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[YOLO-SEG] total time: {total_time:.3f}s")
    print(f"[YOLO-SEG] FPS: {fps:.2f}")


if __name__ == "__main__":
    main()