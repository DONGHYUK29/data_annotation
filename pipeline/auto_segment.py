"""YOLO-seg만으로 1차 세그먼트·YOLO-seg 형식 라벨 생성."""
from __future__ import annotations

import argparse
import json
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
    Edit/CLI에서 전달된 weights 값을 실제 .pt 경로로 변환한다.
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


def mask_to_bbox_and_polygon_norm(mask_u8: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
    m = (mask_u8 > 127).astype(np.uint8)
    h, w = m.shape[:2]
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, epsilon=0.5, closed=True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        approx = cnt.reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        return None, None

    denom_w = max(w - 1, 1)
    denom_h = max(h - 1, 1)
    poly_norm: list[float] = []
    for x, y in approx:
        poly_norm.append(float(np.clip(x / denom_w, 0.0, 1.0)))
        poly_norm.append(float(np.clip(y / denom_h, 0.0, 1.0)))

    xs_norm = poly_norm[0::2]
    ys_norm = poly_norm[1::2]
    bbox_xywhn = [
        (min(xs_norm) + max(xs_norm)) / 2,
        (min(ys_norm) + max(ys_norm)) / 2,
        max(xs_norm) - min(xs_norm),
        max(ys_norm) - min(ys_norm),
    ]
    return bbox_xywhn, poly_norm


def create_overlay(image: np.ndarray, mask: np.ndarray):
    colored = np.zeros_like(image)
    colored[:, :, 1] = mask
    return cv2.addWeighted(image, 1.0, colored, 0.5, 0)


def prediction_color(idx: int) -> tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (29, 185, 84),
        (0, 119, 182),
        (255, 183, 3),
        (131, 56, 236),
        (255, 112, 67),
        (0, 150, 136),
        (244, 67, 154),
    )
    return palette[idx % len(palette)]


def draw_prediction_label(
    rgba: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = rgba.shape[:2]
    x = max(0, min(x, max(0, w - tw - 8)))
    y = max(th + 8, min(y, h - baseline - 4))
    cv2.rectangle(rgba, (x, y - th - 8), (x + tw + 8, y + baseline + 4), (*color, 230), -1)
    cv2.putText(rgba, text, (x + 4, y - 3), font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)


def add_prediction_mask(rgba: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    on = mask > 0
    if not np.any(on):
        return

    alpha = 115
    existing = rgba[..., 3].astype(np.float32) / 255.0
    new_alpha = alpha / 255.0
    out_alpha = new_alpha + existing * (1.0 - new_alpha)

    for ch, value in enumerate(color):
        src = rgba[..., ch].astype(np.float32) / 255.0
        dst = value / 255.0
        blended = np.where(
            out_alpha > 0,
            (dst * new_alpha + src * existing * (1.0 - new_alpha)) / np.maximum(out_alpha, 1e-6),
            0,
        )
        rgba[..., ch] = np.where(on, np.clip(blended * 255.0, 0, 255), rgba[..., ch]).astype(np.uint8)
    rgba[..., 3] = np.where(on, np.clip(out_alpha * 255.0, 0, 255), rgba[..., 3]).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 127
    bb = b > 127
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(aa, bb).sum()
    return float(inter / union)


def collect_prediction_records(results, h: int, w: int) -> list[dict]:
    if results.boxes is None or results.masks is None or results.masks.data is None:
        return []

    boxes = results.boxes
    mask_data = results.masks.data.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else None
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros(len(boxes))
    classes = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(boxes), dtype=int)
    names = getattr(results, "names", {}) or {}

    records: list[dict] = []
    for raw_idx in range(min(len(mask_data), len(boxes))):
        mask = (mask_data[raw_idx] > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if xyxy is not None and raw_idx < len(xyxy):
            x1, y1, x2, y2 = xyxy[raw_idx].astype(float).tolist()
        else:
            ys, xs = np.where(mask > 0)
            x1 = float(xs.min()) if len(xs) else 0.0
            y1 = float(ys.min()) if len(ys) else 0.0
            x2 = float(xs.max()) if len(xs) else 0.0
            y2 = float(ys.max()) if len(ys) else 0.0

        cls_id = int(classes[raw_idx])
        records.append(
            {
                "raw_index": raw_idx,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                "confidence": float(confs[raw_idx]),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "mask": mask,
            }
        )
    return records


def suppress_duplicate_predictions(records: list[dict], iou_threshold: float = 0.75) -> list[dict]:
    kept: list[dict] = []
    for rec in sorted(records, key=lambda r: float(r["confidence"]), reverse=True):
        duplicate = False
        for kept_rec in kept:
            if int(rec["class_id"]) != int(kept_rec["class_id"]):
                continue
            if mask_iou(rec["mask"], kept_rec["mask"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(rec)
    kept.sort(key=lambda r: int(r["raw_index"]))
    return kept


def select_prediction_record(records: list[dict], target_class_id: int | None = None) -> tuple[dict | None, str]:
    if not records:
        return None, "no_predictions"

    if target_class_id is not None:
        matching = [r for r in records if int(r["class_id"]) == int(target_class_id)]
        if matching:
            return max(matching, key=lambda r: float(r["confidence"])), "target_class"

    return max(records, key=lambda r: float(r["confidence"])), "fallback_top_conf"


def save_prediction_overlay_and_meta(
    overlay_path: Path,
    meta_path: Path,
    mask_dir: Path,
    img: np.ndarray,
    records: list[dict],
    conf_threshold: float,
    selected_raw_idx: int | None = None,
    assigned_class_id: int | None = None,
    selection_reason: str | None = None,
    raw_prediction_count: int = 0,
    duplicate_iou_threshold: float = 0.75,
) -> int:
    h, w = img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    meta = {
        "width": w,
        "height": h,
        "threshold": float(conf_threshold),
        "selected_prediction_index": None,
        "selected_raw_index": selected_raw_idx,
        "assigned_class_id": assigned_class_id,
        "selection_reason": selection_reason,
        "raw_prediction_count": int(raw_prediction_count),
        "kept_prediction_count": len(records),
        "duplicate_mask_iou_threshold": float(duplicate_iou_threshold),
        "predictions": [],
    }
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = overlay_path.stem
    for old_mask in mask_dir.glob(f"{stem}_*.png"):
        old_mask.unlink()

    if not records:
        write_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(overlay_path), write_img)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    for idx, rec in enumerate(records):
        mask = rec["mask"]
        is_selected = selected_raw_idx is not None and int(rec["raw_index"]) == int(selected_raw_idx)
        if is_selected:
            meta["selected_prediction_index"] = idx

        mask_name = f"{stem}_{idx:03d}.png"
        mask_path = mask_dir / mask_name
        cv2.imwrite(str(mask_path), mask)

        color = prediction_color(idx)
        if is_selected:
            color = (255, 255, 0)
        add_prediction_mask(rgba, mask, color)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgba, contours, -1, (*color, 255), 2, cv2.LINE_AA)

        cls_id = int(rec["class_id"])
        cls_name = str(rec["class_name"])
        score = float(rec["confidence"])
        x1, y1, x2, y2 = rec["bbox_xyxy"]

        draw_prediction_label(
            rgba,
            f"{cls_name} {score:.2f}",
            int(x1),
            int(y1),
            color,
        )
        meta["predictions"].append(
            {
                "raw_index": int(rec["raw_index"]),
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": score,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "mask_path": str(mask_path),
                "mask_file": mask_name,
                "mask_area_px": int(np.count_nonzero(mask)),
                "selected": is_selected,
            }
        )

    write_img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(overlay_path), write_img)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return count


@torch.inference_mode()
def run_segmentation(
    input_dir: Path | None = None,
    out_dir: Path | None = None,
    yolo_weight: Path | None = None,
    conf: float | None = None,
    batch_size: int | None = None,
) -> None:
    input_dir = input_dir or cfg.IMAGES_RAW_DIR
    out_root = out_dir or cfg.OUTPUT1_DIR

    mask_dir = out_root / "masks"
    label_dir = out_root / "labels"
    overlay_dir = out_root / "images"
    prediction_dir = out_root / "predictions"
    prediction_mask_dir = prediction_dir / "masks"

    mask_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_mask_dir.mkdir(parents=True, exist_ok=True)

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
            stem = img_path.stem
            class_id = get_class_id_from_stem(stem)
            h, w = img.shape[:2]
            raw_records = collect_prediction_records(results, h, w)
            duplicate_iou_threshold = 0.75
            records = suppress_duplicate_predictions(raw_records, duplicate_iou_threshold)
            selected_record, selection_reason = select_prediction_record(records, class_id)
            selected_raw_idx = None if selected_record is None else int(selected_record["raw_index"])

            prediction_overlay_path = prediction_dir / f"{stem}.png"
            prediction_meta_path = prediction_dir / f"{stem}.json"
            save_prediction_overlay_and_meta(
                prediction_overlay_path,
                prediction_meta_path,
                prediction_mask_dir,
                img,
                records,
                conf,
                selected_raw_idx=selected_raw_idx,
                assigned_class_id=class_id,
                selection_reason=selection_reason,
                raw_prediction_count=len(raw_records),
                duplicate_iou_threshold=duplicate_iou_threshold,
            )

            if selected_record is None:
                continue

            detect_count += 1

            mask = selected_record["mask"]
            if mask is None:
                continue

            bbox_xywhn, poly_norm = mask_to_bbox_and_polygon_norm(mask)
            if poly_norm is None or bbox_xywhn is None:
                continue

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
