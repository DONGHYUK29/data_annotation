"""YOLO-seg만으로 1차 세그먼트·YOLO-seg 형식 라벨 생성."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def list_images(input_dir: Path) -> list[Path]:
    imgs: list[Path] = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
        imgs += list(input_dir.glob(ext))
    return sorted(imgs)


def batch_list(data: list, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def get_class_id_from_stem(stem: str) -> int:
    x = int(stem.split("_")[0])
    return x


def mask_to_polygon(mask: np.ndarray):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
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


def normalize_polygon(poly, w: int, h: int):
    poly_out = []
    denom_w = max(w - 1, 1)
    denom_h = max(h - 1, 1)

    for x, y in poly:
        poly_out.append(float(x) / denom_w)
        poly_out.append(float(y) / denom_h)

    return poly_out


def save_label(path: Path, class_id: int, poly_norm: list[float]) -> None:
    line = str(class_id) + "".join(f" {v:.6f}" for v in poly_norm)
    path.write_text(line + "\n", encoding="utf-8")


def create_overlay(image: np.ndarray, mask: np.ndarray):
    colored = np.zeros_like(image)
    colored[:, :, 1] = mask
    return cv2.addWeighted(image, 1.0, colored, 0.5, 0)


def select_best_mask(results):
    """
    YOLO-seg 결과에서 confidence가 가장 높은 instance 하나의 mask를 선택.
    반환:
        mask_uint8: (H, W), 값 0 또는 255
        pred_cls_id: 예측 클래스 id (없으면 None)
    """
    if results.boxes is None or len(results.boxes) == 0:
        return None, None

    if results.masks is None or results.masks.data is None:
        return None, None

    scores = results.boxes.conf.detach().cpu().numpy()
    if len(scores) == 0:
        return None, None

    best_idx = int(np.argmax(scores))

    masks = results.masks.data.detach().cpu().numpy()  # [N, H, W]
    if best_idx >= len(masks):
        return None, None

    mask = (masks[best_idx] > 0.5).astype(np.uint8) * 255

    pred_cls_id = None
    if results.boxes.cls is not None and len(results.boxes.cls) > best_idx:
        pred_cls_id = int(results.boxes.cls[best_idx].item())

    return mask, pred_cls_id


@torch.inference_mode()
def run_segmentation(
    input_dir: Path | None = None,
    out_dir: Path | None = None,
    yolo_weight: Path | None = None,
    conf: float | None = None,
    batch_size: int | None = None,
) -> None:
    input_dir = input_dir or cfg.IMAGES_RAW_DIR
    out_root = out_dir or cfg.STAGE1_DIR
    mask_dir = out_root / "masks"
    label_dir = out_root / "labels"
    overlay_dir = out_root / "images"
    mask_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    yw = yolo_weight or cfg.YOLO_WEIGHT
    conf = cfg.SEG_CONF_THRESHOLD if conf is None else conf
    batch_size = cfg.SEG_BATCH_SIZE if batch_size is None else batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading YOLO-seg...")
    yolo = YOLO(str(yw))

    img_paths = list_images(input_dir)
    print("Total images:", len(img_paths))

    detect_count = 0
    save_count = 0

    for batch_paths in tqdm(list(batch_list(img_paths, batch_size)), desc="YOLO batches"):
        images = []
        valid_paths: list[Path] = []

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
            conf=conf,
            verbose=False,
            device=0 if device == "cuda" else "cpu",
        )

        for img_path, img, results in zip(valid_paths, images, results_list):
            h, w = img.shape[:2]

            mask, pred_cls_id = select_best_mask(results)
            if mask is None:
                continue

            detect_count += 1

            poly = mask_to_polygon(mask)
            if poly is None:
                continue

            poly_norm = normalize_polygon(poly, w, h)
            if len(poly_norm) < 6:
                continue

            stem = img_path.stem

            class_id = get_class_id_from_stem(stem)


            mask_path = mask_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.txt"
            overlay_path = overlay_dir / f"{stem}.png"

            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(overlay_path), create_overlay(img, mask))
            save_label(label_path, class_id, poly_norm)
            save_count += 1

    print("\nSUMMARY")
    print("Detected:", detect_count)
    print("Saved:", save_count)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YOLO-seg만으로 1차 마스킹")
    parser.add_argument("--input", type=Path, default=None, help="원본 이미지 폴더")
    parser.add_argument("--output", type=Path, default=None, help="output_1 에 해당하는 루트")
    args = parser.parse_args(argv)
    run_segmentation(input_dir=args.input, out_dir=args.output)


if __name__ == "__main__":
    main()