import os

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm


# =========================
# Paths
# =========================

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(
    os.environ.get("KERI_ANNOTATION_WORKDIR", REPO_ROOT / "data" / "annotation_work")
).expanduser().resolve()

INPUT_DIR = BASE_DIR / "input"

OUTPUT_DIR = BASE_DIR / "output_1"
MASK_DIR = OUTPUT_DIR / "masks"
LABEL_DIR = OUTPUT_DIR / "labels"
OVERLAY_DIR = OUTPUT_DIR / "images"

MASK_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Weights
# =========================

YOLO_WEIGHT = str(
    Path(os.environ.get("YOLO_DETECTION_MODEL", REPO_ROOT / "weights" / "yolo26x.pt"))
    .expanduser()
    .resolve()
)
SAM_WEIGHT = str(
    Path(os.environ.get("SAM_CHECKPOINT", REPO_ROOT / "weights" / "sam_vit_b_01ec64.pth"))
    .expanduser()
    .resolve()
)


# =========================
# Params
# =========================

CONF_THRES = 0.01
BBOX_EXPAND = 1.15
BATCH_SIZE = 16


# =========================
# Device
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# Models
# =========================

print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHT)

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_WEIGHT)
sam.to(device)

predictor = SamPredictor(sam)


# =========================
# Utils
# =========================

def list_images():

    imgs = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
        imgs += list(INPUT_DIR.glob(ext))

    return sorted(imgs)


def batch_list(data, size):

    for i in range(0, len(data), size):
        yield data[i:i+size]


def get_class_id(stem):

    x = int(stem.split("_")[0])
    return x - 1


def expand_bbox(box, w, h):

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = (x2 - x1) * BBOX_EXPAND
    bh = (y2 - y1) * BBOX_EXPAND

    nx1 = max(0, cx - bw / 2)
    ny1 = max(0, cy - bh / 2)
    nx2 = min(w - 1, cx + bw / 2)
    ny2 = min(h - 1, cy + bh / 2)

    return np.array([nx1, ny1, nx2, ny2])


def mask_to_polygon(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 10:
        return None

    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.002 * perimeter

    approx = cv2.approxPolyDP(cnt, epsilon, True)

    poly = approx.reshape(-1, 2)

    if len(poly) < 3:
        return None

    return poly.astype(np.float32)


def normalize_polygon(poly, w, h):

    poly_out = []

    for x, y in poly:

        xn = x / (w - 1)
        yn = y / (h - 1)

        poly_out.append(xn)
        poly_out.append(yn)

    return poly_out


def save_label(path, class_id, poly):

    line = str(class_id)

    for v in poly:
        line += f" {v:.6f}"

    with open(path, "w") as f:
        f.write(line + "\n")


def create_overlay(image, mask):

    colored = np.zeros_like(image)

    colored[:, :, 1] = mask

    overlay = cv2.addWeighted(image, 1.0, colored, 0.5, 0)

    return overlay


# =========================
# Main
# =========================

@torch.inference_mode()
def main():

    img_paths = list_images()

    print("Total images:", len(img_paths))

    detect_count = 0
    save_count = 0

    for batch_paths in tqdm(list(batch_list(img_paths, BATCH_SIZE)), desc="YOLO batches"):

        images = []
        valid_paths = []

        for p in batch_paths:

            img = cv2.imread(str(p))

            if img is None:
                continue

            images.append(img)
            valid_paths.append(p)

        if not images:
            continue

        results_list = yolo(
            images,
            conf=CONF_THRES,
            verbose=False,
            device=0 if device == "cuda" else "cpu"
        )

        for img_path, img, results in zip(valid_paths, images, results_list):

            h, w = img.shape[:2]

            if results.boxes is None or len(results.boxes) == 0:
                continue

            detect_count += 1

            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()

            box = boxes[np.argmax(scores)]

            box = expand_bbox(box, w, h)

            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )

            mask = (masks[0] > 0.5).astype(np.uint8) * 255

            poly = mask_to_polygon(mask)

            if poly is None:
                continue

            poly_norm = normalize_polygon(poly, w, h)

            if len(poly_norm) < 6:
                continue

            stem = img_path.stem
            class_id = get_class_id(stem)

            mask_path = MASK_DIR / f"{stem}.png"
            label_path = LABEL_DIR / f"{stem}.txt"
            overlay_path = OVERLAY_DIR / f"{stem}.png"

            cv2.imwrite(str(mask_path), mask)

            overlay = create_overlay(img, mask)
            cv2.imwrite(str(overlay_path), overlay)

            save_label(label_path, class_id, poly_norm)

            save_count += 1


    print("\nSUMMARY")
    print("Detected:", detect_count)
    print("Saved:", save_count)


if __name__ == "__main__":
    main()
