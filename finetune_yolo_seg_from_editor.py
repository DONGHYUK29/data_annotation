# finetune_yolo_seg_from_editor.py
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent  # new training repo root OR data_annotation root
DATA_ANN_DIR = BASE_DIR  # 이 스크립트를 data_annotation 루트에 두면 그대로 사용 가능

LABELS_IN_DIR = DATA_ANN_DIR / "images" / "edited_results" / "labels"
MASKS_IN_DIR  = DATA_ANN_DIR / "images" / "edited_results" / "masks"

# RGB 이미지 탐색 후보
ORIGINAL_IMG_DIR = DATA_ANN_DIR / "images" / "original"
EDITED_IMG_DIR   = DATA_ANN_DIR / "images" / "edited_results" / "images"  # 있으면 사용

WEIGHTS_DIR = DATA_ANN_DIR / "weights"
BASE_WEIGHTS = WEIGHTS_DIR / "yolo26l-seg.pt"

# 결과 저장 이름(원하는 파일명으로 바꿔)
OUTPUT_WEIGHTS_NAME = "yolo26l-seg-finetuned-7cls.pt"

# YOLO dataset output
DATASET_DIR = DATA_ANN_DIR / "datasets" / "finetune7seg"
TRAIN_RATIO = 0.8
IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

# 7 classes (순서가 class_id 0~6)
CLASS_NAMES = [
    "class0",
    "class1",
    "class2",
    "class3",
    "class4",
    "class5",
    "class6",
]

# Train settings
IMGSZ = 640
EPOCHS = 50
BATCH = 8
SEED = 42


# =========================
# Utils
# =========================
def find_image_by_stem(stem: str) -> Optional[Path]:
    """Find RGB image file by stem in known locations."""
    for d in (ORIGINAL_IMG_DIR, EDITED_IMG_DIR):
        if not d.exists():
            continue
        for ext in IMG_EXTS:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def parse_editor_label_file(label_path: Path) -> List[Tuple[str, int, List[float]]]:
    """
    Parse custom label file lines:
      instance_id class_id x1 y1 x2 y2 poly_x0 poly_y0 ...
    Return list of (instance_id, class_id, polygon_xy_pixels_flat)
    polygon list may be empty if not present.
    """
    out = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        instance_id = parts[0]
        cls_id = int(float(parts[1]))

        # expected: inst, cls, x1 y1 x2 y2, then polygon pairs...
        poly = []
        if len(parts) > 6:
            # parts[2:6] = bbox, parts[6:] polygon
            try:
                poly = [float(x) for x in parts[6:]]
            except ValueError:
                poly = []
        out.append((instance_id, cls_id, poly))
    return out


def mask_to_polygon(mask_u8: np.ndarray, min_points: int = 6) -> List[float]:
    """
    Convert binary mask to a polygon (pixel coords flat list x0 y0 x1 y1 ...)
    Uses largest contour. Approximates contour to reduce points.
    """
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    _, th = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    contours, _hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)

    # approximate
    peri = cv2.arcLength(cnt, True)
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)

    pts = approx.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return []

    # ensure enough points (optional): if too few, fall back to raw contour sampling
    if len(pts) < (min_points // 2):
        pts = cnt.reshape(-1, 2).astype(float)
        # sample down if huge
        if len(pts) > 200:
            idx = np.linspace(0, len(pts) - 1, 200).astype(int)
            pts = pts[idx]

    poly = []
    for x, y in pts:
        poly.extend([float(x), float(y)])
    return poly


def normalize_poly(poly_xy: List[float], w: int, h: int) -> List[float]:
    """Normalize pixel coords to [0,1]."""
    if not poly_xy:
        return []
    out = []
    for i in range(0, len(poly_xy), 2):
        x = poly_xy[i] / max(w, 1)
        y = poly_xy[i + 1] / max(h, 1)
        # clip
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        y = 0.0 if y < 0 else (1.0 if y > 1 else y)
        out.extend([x, y])
    return out


def image_id_from_label_path(label_path: Path) -> str:
    """
    Example: 1_edited.txt -> image_id "1"
    If your naming differs, adjust here.
    """
    stem = label_path.stem
    if stem.endswith("_edited"):
        return stem[:-7]
    return stem


def collect_instances() -> Dict[str, List[Tuple[int, List[float]]]]:
    """
    Returns dict:
      image_id -> list of (class_id, polygon_xy_pixels_flat)
    Polygon may be empty initially; later we can fill from masks.
    """
    items: Dict[str, List[Tuple[int, List[float]]]] = {}

    for lp in sorted(LABELS_IN_DIR.glob("*.txt")):
        img_id = image_id_from_label_path(lp)
        parsed = parse_editor_label_file(lp)
        if not parsed:
            continue
        items.setdefault(img_id, [])
        for instance_id, cls_id, poly in parsed:
            items[img_id].append((cls_id, poly))
    return items


def find_mask_for_instance(image_id: str, inst_index: int) -> Optional[Path]:
    """
    Try to find mask file for a given image_id and object index.
    Typical pattern: {image_id}_obj{idx}_masked_edited.png
    Adjust if your naming differs.
    """
    cand = MASKS_IN_DIR / f"{image_id}_obj{inst_index}_masked_edited.png"
    if cand.exists():
        return cand

    # fallback: search with prefix
    matches = sorted(MASKS_IN_DIR.glob(f"{image_id}_obj{inst_index}_*edited.png"))
    return matches[0] if matches else None


def ensure_dataset_dirs():
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    names_yaml = "\n".join([f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    yaml_path.write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )


def prepare_dataset():
    random.seed(SEED)
    ensure_dataset_dirs()
    write_data_yaml()

    items = collect_instances()
    img_ids = sorted(items.keys())
    if not img_ids:
        raise RuntimeError(f"No label txt found in: {LABELS_IN_DIR}")

    random.shuffle(img_ids)
    n_train = max(1, int(len(img_ids) * TRAIN_RATIO))
    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train:]) if len(img_ids) > n_train else set(img_ids[:1])

    for img_id in img_ids:
        split = "train" if img_id in train_ids else "val"

        # Find RGB image
        img_path = find_image_by_stem(img_id)
        img_bgr = None
        if img_path is not None:
            img_bgr = cv2.imread(str(img_path))
        else:
            # fallback: create dummy image from first available mask
            m0 = find_mask_for_instance(img_id, 0)
            if m0 is None:
                print(f"[SKIP] No image and no mask for image_id={img_id}")
                continue
            mu = cv2.imread(str(m0), cv2.IMREAD_GRAYSCALE)
            if mu is None:
                print(f"[SKIP] Failed to read mask: {m0}")
                continue
            img_bgr = cv2.cvtColor(mu, cv2.COLOR_GRAY2BGR)

        h, w = img_bgr.shape[:2]

        # Copy image into dataset
        out_img = DATASET_DIR / "images" / split / f"{img_id}.png"
        cv2.imwrite(str(out_img), img_bgr)

        # Build YOLO-seg label lines
        lines = []
        for inst_idx, (cls_id, poly_pix) in enumerate(items[img_id]):
            poly = poly_pix

            # If no polygon in txt, derive from mask
            if not poly or len(poly) < 6:
                mp = find_mask_for_instance(img_id, inst_idx)
                if mp and mp.exists():
                    mu = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if mu is not None:
                        poly = mask_to_polygon(mu)
                # still empty -> skip
            if not poly or len(poly) < 6:
                continue

            poly_norm = normalize_poly(poly, w, h)
            if len(poly_norm) < 6:
                continue

            # YOLO-seg line: class_id + poly coords
            line = " ".join([str(int(cls_id))] + [f"{v:.6f}" for v in poly_norm])
            lines.append(line)

        out_lbl = DATASET_DIR / "labels" / split / f"{img_id}.txt"
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"[OK] Dataset prepared at: {DATASET_DIR}")
    print(f" - train images: {len(list((DATASET_DIR/'images'/'train').glob('*')))}")
    print(f" - val images:   {len(list((DATASET_DIR/'images'/'val').glob('*')))}")


def finetune():
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError(f"Base weights not found: {BASE_WEIGHTS}")

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(str(BASE_WEIGHTS))
    
    device = 0 if torch.cuda.is_available() else "cpu"

    # Train
    results = model.train(
        task="segment",
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        seed=SEED,
        device=device,
        workers=4,
        pretrained=True,
        val=False,
    )

    # Find best.pt
    # Ultralytics outputs in runs/segment/train*/weights/best.pt by default
    runs_dir = Path(results.save_dir)
    best_pt = runs_dir / "weights" / "best.pt"
    if not best_pt.exists():
        # fallback: search
        candidates = list(Path("runs").rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError("best.pt not found after training.")
        best_pt = max(candidates, key=lambda p: p.stat().st_mtime)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_pt = WEIGHTS_DIR / OUTPUT_WEIGHTS_NAME
    shutil.copy2(best_pt, out_pt)

    print(f"[OK] Saved finetuned weights: {out_pt}")
    print(f"[INFO] best.pt source: {best_pt}")


def main():
    print("[START] main() called")
    prepare_dataset()
    finetune()

if __name__ == "__main__":
    main()