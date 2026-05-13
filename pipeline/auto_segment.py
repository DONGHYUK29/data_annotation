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
    return int(stem.split("_")[0])


def resolve_yolo_weight(weights_arg: str | None) -> Path:
    """
    GUI/CLI에서 전달된 weights 값을 실제 .pt 경로로 변환한다.
    - 빈 값이면 config.py의 cfg.YOLO_WEIGHT 사용
    - 상대 경로이면 cfg.WEIGHTS_DIR 하위로 해석
    - 절대 경로/상대 경로 모두 최종적으로 cfg.WEIGHTS_DIR 내부 .pt만 허용
    """
    if weights_arg is None or str(weights_arg).strip() == "":
        return Path(cfg.YOLO_WEIGHT)

    wd = Path(cfg.WEIGHTS_DIR).resolve()
    raw = Path(str(weights_arg).strip()).expanduser()

    if raw.suffix.lower() != ".pt":
        raise FileNotFoundError(f"세그먼트 가중치는 .pt 파일만 허용합니다: {weights_arg!r}")

    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (wd / raw).resolve()

    try:
        candidate.relative_to(wd)
    except ValueError:
        raise FileNotFoundError(f"세그먼트 가중치는 weights 폴더 내부 파일만 사용할 수 있습니다: {wd}") from None

    if not candidate.is_file():
        raise FileNotFoundError(f"세그먼트 가중치 파일을 찾을 수 없습니다: {candidate}")

    return candidate


def save_label(
    path: Path,
    class_id: int,
    bbox_xywhn: list[float],
    poly_norm: list[float],
) -> None:
    """
    저장 형식:
    class xc yc w h x1 y1 x2 y2 ...
    """
    if len(bbox_xywhn) != 4:
        raise ValueError(f"bbox_xywhn must have length 4, got {len(bbox_xywhn)}")

    xc, yc, w, h = [float(v) for v in bbox_xywhn]
    values = [class_id, xc, yc, w, h, *[float(v) for v in poly_norm]]
    line = str(values[0]) + "".join(f" {float(v):.6f}" for v in values[1:])
    path.write_text(line + "\n", encoding="utf-8")


def create_overlay(image: np.ndarray, mask: np.ndarray):
    colored = np.zeros_like(image)
    colored[:, :, 1] = mask
    return cv2.addWeighted(image, 1.0, colored, 0.5, 0)


def select_best_instance(results):
    if results.boxes is None or len(results.boxes) == 0:
        return None, None

    if results.masks is None or results.masks.data is None:
        return None, None

    scores = results.boxes.conf.detach().cpu().numpy()
    if len(scores) == 0:
        return None, None

    best_idx = int(np.argmax(scores))

    pred_cls_id = None
    if results.boxes.cls is not None and len(results.boxes.cls) > best_idx:
        pred_cls_id = int(results.boxes.cls[best_idx].item())

    return best_idx, pred_cls_id


def get_mask_polygon_and_bbox_from_results(results, best_idx: int):
    if results.masks is None or results.masks.data is None:
        return None, None, None

    if results.boxes is None or len(results.boxes) == 0:
        return None, None, None

    if best_idx >= len(results.masks.data):
        return None, None, None

    if results.boxes.xywhn is None or len(results.boxes.xywhn) <= best_idx:
        return None, None, None

    mask = results.masks.data[best_idx].detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    if not hasattr(results.masks, "xyn"):
        return mask, None, None

    segs = results.masks.xyn
    if segs is None or best_idx >= len(segs):
        return mask, None, None

    poly = segs[best_idx]
    if poly is None or len(poly) < 3:
        return mask, None, None

    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    poly_norm: list[float] = []
    for x, y in poly:
        poly_norm.append(float(x))
        poly_norm.append(float(y))

    if len(poly_norm) < 6:
        return mask, None, None

    bbox_xywhn = (
        results.boxes.xywhn[best_idx]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(-1)
        .tolist()
    )

    if len(bbox_xywhn) != 4:
        return mask, poly_norm, None

    return mask, poly_norm, bbox_xywhn


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
            retina_masks=True,
        )

        for img_path, img, results in zip(valid_paths, images, results_list):
            best_idx, pred_cls_id = select_best_instance(results)
            if best_idx is None:
                continue

            detect_count += 1

            mask, poly_norm, bbox_xywhn = get_mask_polygon_and_bbox_from_results(
                results, best_idx
            )
            if mask is None or poly_norm is None or bbox_xywhn is None:
                continue

            stem = img_path.stem
            class_id = get_class_id_from_stem(stem)

            mask_path = mask_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.txt"
            overlay_path = overlay_dir / f"{stem}.png"

            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(overlay_path), create_overlay(img, mask))
            save_label(label_path, class_id, bbox_xywhn, poly_norm)

            save_count += 1

    print("\nSUMMARY")
    print("Detected:", detect_count)
    print("Saved:", save_count)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="YOLO-seg만으로 1차 마스킹")
    parser.add_argument("--input", type=Path, default=None, help="원본 이미지 폴더")
    parser.add_argument("--output", type=Path, default=None, help="output_1 에 해당하는 루트")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="weights/ 하위 .pt 경로 (예: exp1/weights/best.pt). 비우면 cfg.YOLO_WEIGHT 사용",
    )
    args = parser.parse_args(argv)
    run_segmentation(
        input_dir=args.input,
        out_dir=args.output,
        yolo_weight=resolve_yolo_weight(args.weights),
    )


if __name__ == "__main__":
    main()