#!/usr/bin/env python3
"""Evaluate YOLO segmentation with optional depth-based mask refinement.

This script is intentionally independent of ROS. It uses the same high-level
idea as the realtime depth refine mode: run the RGB segmentation model first,
then refine the predicted masks using an aligned depth image before calculating
mask AP.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = (".png", ".tif", ".tiff", ".npy", ".npz", ".depth", ".exr")


@dataclass
class EvalParams:
    depth_scale: float = 0.001
    depth_valid_min_m: float = 0.10
    depth_valid_max_m: float = 3.00
    depth_refine_abs_delta_m: float = 0.06
    depth_refine_rel_delta: float = 0.08
    depth_refine_erode_kernel: int = 5
    depth_refine_boundary_iter: int = 2
    depth_refine_close_kernel: int = 3
    depth_refine_bottom_keep_ratio: float = 0.30
    depth_refine_min_area_ratio: float = 0.85
    depth_refine_min_mask_area: int = 80
    depth_refine_min_core_pixels: int = 30
    depth_refine_min_valid_ratio: float = 0.20
    depth_refine_keep_largest_component: bool = True


@dataclass
class Instance:
    image_id: int
    cls: int
    mask: np.ndarray
    conf: float = 1.0


def make_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def load_dataset_yaml(path: Path) -> Tuple[Path, Path, Path, Dict[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", path.parent)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    val_entry = data.get("val", data.get("test", "images/val"))
    image_dir = Path(val_entry).expanduser()
    if not image_dir.is_absolute():
        image_dir = root / image_dir

    if "images" in image_dir.parts:
        parts = list(image_dir.parts)
        idx = len(parts) - 1 - parts[::-1].index("images")
        label_dir = Path(*parts[:idx], "labels", *parts[idx + 1 :])
    else:
        label_dir = root / "labels" / image_dir.name

    names_raw = data.get("names", {})
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        names = {i: str(v) for i, v in enumerate(names_raw)}
    return root, image_dir, label_dir, names


def list_images(image_dir: Path, limit: int = 0) -> List[Path]:
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit > 0 else images


def label_path_for_image(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    rel = image_path.relative_to(image_dir)
    return (label_dir / rel).with_suffix(".txt")


def load_yolo_seg_labels(label_path: Path, hw: Tuple[int, int], image_id: int) -> List[Instance]:
    h, w = hw
    instances: List[Instance] = []
    if not label_path.exists():
        return instances

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        if coords.size < 6 or coords.size % 2 != 0:
            continue
        pts = coords.reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
        poly = np.round(pts).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        if int(mask.sum()) > 0:
            instances.append(Instance(image_id=image_id, cls=cls, mask=mask))
    return instances


def resize_mask(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    if mask.shape[:2] == (h, w):
        return (mask > 0.5).astype(np.uint8)
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)


def predict_instances(
    model: YOLO,
    image_path: Path,
    image_id: int,
    hw: Tuple[int, int],
    args: argparse.Namespace,
) -> List[Instance]:
    result = model.predict(
        source=str(image_path),
        task="segment",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )[0]

    if result.boxes is None or result.masks is None:
        return []

    boxes = result.boxes
    cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    masks = result.masks.data.detach().cpu().numpy()

    preds: List[Instance] = []
    for i in range(min(len(cls), len(masks))):
        mask = resize_mask(masks[i], hw)
        if int(mask.sum()) <= 0:
            continue
        preds.append(
            Instance(
                image_id=image_id,
                cls=int(cls[i]),
                mask=mask,
                conf=float(conf[i]),
            )
        )
    return preds


def find_depth_path(
    image_path: Path,
    image_dir: Path,
    depth_root: Optional[Path],
    pattern: str,
) -> Optional[Path]:
    if depth_root is None:
        return None
    rel = image_path.relative_to(image_dir)
    values = {
        "stem": image_path.stem,
        "name": image_path.name,
        "suffix": image_path.suffix,
        "rel": rel.as_posix(),
        "rel_stem": rel.with_suffix("").as_posix(),
        "parent": rel.parent.as_posix() if rel.parent.as_posix() != "." else "",
    }

    candidate = depth_root / pattern.format(**values)
    if candidate.exists():
        return candidate

    stem = image_path.stem
    for ext in DEPTH_EXTS:
        candidate = depth_root / rel.with_suffix(ext)
        if candidate.exists():
            return candidate
        candidate = depth_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_depth_m(depth_path: Path, image_hw: Tuple[int, int], depth_scale: float) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix in (".npy", ".depth"):
        depth = np.load(depth_path)
    elif suffix == ".npz":
        npz = np.load(depth_path)
        key = "depth" if "depth" in npz.files else npz.files[0]
        depth = npz[key]
    else:
        flags = cv2.IMREAD_UNCHANGED
        depth = cv2.imread(str(depth_path), flags)
        if depth is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    if depth_path.suffix.lower() != ".exr" and depth.max(initial=0.0) > 20.0:
        depth = depth * float(depth_scale)

    h, w = image_hw
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


def valid_depth_mask(depth_m: np.ndarray, params: EvalParams) -> np.ndarray:
    return (
        np.isfinite(depth_m)
        & (depth_m > params.depth_valid_min_m)
        & (depth_m < params.depth_valid_max_m)
    )


def clamp_bbox(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), width - 1))
    x2 = max(0, min(int(x2), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    return (labels == largest_label).astype(np.uint8)


def refine_mask_with_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    bbox: Tuple[int, int, int, int],
    params: EvalParams,
) -> Tuple[np.ndarray, bool, str]:
    h_img, w_img = mask.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(*bbox, width=w_img, height=h_img)
    roi_mask = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    roi_depth = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    original_area = int(roi_mask.sum())
    if original_area < params.depth_refine_min_mask_area:
        return mask.copy(), False, "small_mask"

    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params.depth_refine_erode_kernel, params.depth_refine_erode_kernel),
    )
    core = cv2.erode(roi_mask, erode_kernel, iterations=1)
    if int(core.sum()) < params.depth_refine_min_core_pixels:
        core = roi_mask.copy()

    valid_depth = valid_depth_mask(roi_depth, params)
    core_valid = (core > 0) & valid_depth
    valid_count = int(core_valid.sum())
    if valid_count < params.depth_refine_min_core_pixels:
        return mask.copy(), False, "not_enough_core_depth"

    valid_ratio = valid_count / max(1, int(core.sum()))
    if valid_ratio < params.depth_refine_min_valid_ratio:
        return mask.copy(), False, "low_valid_depth_ratio"

    z_obj = float(np.median(roi_depth[core_valid]))
    delta = max(params.depth_refine_abs_delta_m, z_obj * params.depth_refine_rel_delta)

    eroded_for_boundary = cv2.erode(
        roi_mask, erode_kernel, iterations=params.depth_refine_boundary_iter
    )
    boundary = (roi_mask > 0) & (eroded_for_boundary == 0)
    roi_h, _ = roi_mask.shape[:2]
    keep_bottom_start_y = int(round(roi_h * (1.0 - params.depth_refine_bottom_keep_ratio)))
    yy = np.arange(roi_h).reshape(-1, 1)
    not_bottom_zone = yy < keep_bottom_start_y
    remove_roi = boundary & not_bottom_zone & valid_depth & (np.abs(roi_depth - z_obj) > delta)

    refined_roi = roi_mask.copy()
    refined_roi[remove_roi] = 0
    if params.depth_refine_close_kernel >= 3:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.depth_refine_close_kernel, params.depth_refine_close_kernel),
        )
        refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_CLOSE, close_kernel)
    if params.depth_refine_keep_largest_component:
        refined_roi = keep_largest_component(refined_roi)

    refined_area = int(refined_roi.sum())
    area_ratio = refined_area / max(1, original_area)
    if refined_area <= 0:
        return mask.copy(), False, "empty_after_refine_reverted"
    if area_ratio < params.depth_refine_min_area_ratio:
        return mask.copy(), False, "too_much_removed_reverted"

    refined_full = mask.copy().astype(np.uint8)
    refined_full[y1 : y2 + 1, x1 : x2 + 1] = refined_roi.astype(np.uint8)
    return refined_full, True, "ok"


def apply_depth_refine(
    preds: Sequence[Instance],
    depth_m: Optional[np.ndarray],
    params: EvalParams,
) -> Tuple[List[Instance], Dict[str, int]]:
    stats: Dict[str, int] = {"total": len(preds), "applied": 0, "missing_depth": 0}
    if depth_m is None:
        stats["missing_depth"] = len(preds)
        return list(preds), stats

    refined: List[Instance] = []
    for pred in preds:
        mask, applied, reason = refine_mask_with_depth(
            pred.mask, depth_m, mask_bbox(pred.mask), params
        )
        stats[reason] = stats.get(reason, 0) + 1
        if applied:
            stats["applied"] += 1
        if int(mask.sum()) > 0:
            refined.append(Instance(pred.image_id, pred.cls, mask, pred.conf))
    return refined, stats


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(inter / max(1, union))


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def evaluate_map(
    preds: Sequence[Instance],
    gts: Sequence[Instance],
    class_ids: Iterable[int],
    thresholds: Sequence[float],
) -> Dict[str, object]:
    gt_by_class: Dict[int, List[Instance]] = {}
    pred_by_class: Dict[int, List[Instance]] = {}
    for gt in gts:
        gt_by_class.setdefault(gt.cls, []).append(gt)
    for pred in preds:
        pred_by_class.setdefault(pred.cls, []).append(pred)

    per_class = {}
    aps = []
    ap50s = []
    for cls in class_ids:
        cls_gts = gt_by_class.get(cls, [])
        if not cls_gts:
            continue
        cls_preds = sorted(pred_by_class.get(cls, []), key=lambda p: p.conf, reverse=True)
        ap_values = []
        for thr in thresholds:
            matched = set()
            tp = np.zeros(len(cls_preds), dtype=np.float32)
            fp = np.zeros(len(cls_preds), dtype=np.float32)

            gts_by_image: Dict[int, List[Tuple[int, Instance]]] = {}
            for idx, gt in enumerate(cls_gts):
                gts_by_image.setdefault(gt.image_id, []).append((idx, gt))

            for i, pred in enumerate(cls_preds):
                candidates = gts_by_image.get(pred.image_id, [])
                best_iou = 0.0
                best_idx = -1
                for gt_idx, gt in candidates:
                    if gt_idx in matched:
                        continue
                    iou = mask_iou(pred.mask, gt.mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_idx
                if best_iou >= thr and best_idx >= 0:
                    tp[i] = 1.0
                    matched.add(best_idx)
                else:
                    fp[i] = 1.0

            if len(cls_preds) == 0:
                ap = 0.0
            else:
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / max(1, len(cls_gts))
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
                ap = compute_ap(recall, precision)
            ap_values.append(ap)

        ap50 = ap_values[0]
        map5095 = float(np.mean(ap_values))
        ap50s.append(ap50)
        aps.append(map5095)
        per_class[int(cls)] = {
            "gt": len(cls_gts),
            "pred": len(cls_preds),
            "mask_mAP50": ap50,
            "mask_mAP50_95": map5095,
        }

    return {
        "mask_mAP50": float(np.mean(ap50s)) if ap50s else 0.0,
        "mask_mAP50_95": float(np.mean(aps)) if aps else 0.0,
        "per_class": per_class,
    }


def merge_stats(total: Dict[str, int], update: Dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RGB-only and depth-refined YOLO segmentation mask mAP."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset.yaml")
    parser.add_argument("--split", default="val", choices=["val"], help="Currently evaluates dataset val path.")
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument(
        "--depth-pattern",
        default="{stem}.png",
        help=(
            "Path pattern relative to --depth-root. Available keys: "
            "{stem}, {name}, {suffix}, {rel}, {rel_stem}, {parent}."
        ),
    )
    parser.add_argument("--mode", default="both", choices=["rgb", "depth", "both"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for number of images.")
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-valid-min-m", type=float, default=0.10)
    parser.add_argument("--depth-valid-max-m", type=float, default=3.00)
    parser.add_argument("--depth-refine-abs-delta-m", type=float, default=0.06)
    parser.add_argument("--depth-refine-rel-delta", type=float, default=0.08)
    parser.add_argument("--depth-refine-erode-kernel", type=int, default=5)
    parser.add_argument("--depth-refine-boundary-iter", type=int, default=2)
    parser.add_argument("--depth-refine-close-kernel", type=int, default=3)
    parser.add_argument("--depth-refine-bottom-keep-ratio", type=float, default=0.30)
    parser.add_argument("--depth-refine-min-area-ratio", type=float, default=0.85)
    parser.add_argument("--depth-refine-min-mask-area", type=int, default=80)
    parser.add_argument("--depth-refine-min-core-pixels", type=int, default=30)
    parser.add_argument("--depth-refine-min-valid-ratio", type=float, default=0.20)
    parser.add_argument("--no-depth-refine-largest-component", action="store_true")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> EvalParams:
    return EvalParams(
        depth_scale=args.depth_scale,
        depth_valid_min_m=args.depth_valid_min_m,
        depth_valid_max_m=args.depth_valid_max_m,
        depth_refine_abs_delta_m=args.depth_refine_abs_delta_m,
        depth_refine_rel_delta=args.depth_refine_rel_delta,
        depth_refine_erode_kernel=make_odd(args.depth_refine_erode_kernel),
        depth_refine_boundary_iter=max(1, args.depth_refine_boundary_iter),
        depth_refine_close_kernel=make_odd(args.depth_refine_close_kernel),
        depth_refine_bottom_keep_ratio=float(np.clip(args.depth_refine_bottom_keep_ratio, 0.0, 0.9)),
        depth_refine_min_area_ratio=float(np.clip(args.depth_refine_min_area_ratio, 0.0, 1.0)),
        depth_refine_min_mask_area=args.depth_refine_min_mask_area,
        depth_refine_min_core_pixels=args.depth_refine_min_core_pixels,
        depth_refine_min_valid_ratio=args.depth_refine_min_valid_ratio,
        depth_refine_keep_largest_component=not args.no_depth_refine_largest_component,
    )


def main() -> int:
    args = parse_args()
    params = params_from_args(args)
    _, image_dir, label_dir, names = load_dataset_yaml(args.data)
    images = list_images(image_dir, args.limit)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    depth_root = args.depth_root.expanduser().resolve() if args.depth_root else None
    if args.mode in ("depth", "both") and depth_root is None:
        raise RuntimeError("--depth-root is required for depth or both mode.")

    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Images: {len(images)} from {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Device: {args.device}")
    if depth_root:
        print(f"Depth root: {depth_root}")
        print(f"Depth pattern: {args.depth_pattern}")

    model = YOLO(str(args.model))
    thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    class_ids = sorted(names.keys())
    all_gts: List[Instance] = []
    rgb_preds: List[Instance] = []
    depth_preds: List[Instance] = []
    depth_stats: Dict[str, int] = {}
    missing_depth_images = 0

    for image_id, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] failed to read image: {image_path}")
            continue
        h, w = image.shape[:2]
        label_path = label_path_for_image(image_path, image_dir, label_dir)
        all_gts.extend(load_yolo_seg_labels(label_path, (h, w), image_id))

        preds = predict_instances(model, image_path, image_id, (h, w), args)
        if args.mode in ("rgb", "both"):
            rgb_preds.extend(preds)

        if args.mode in ("depth", "both"):
            depth_path = find_depth_path(image_path, image_dir, depth_root, args.depth_pattern)
            depth_m = None
            if depth_path is None:
                missing_depth_images += 1
            else:
                try:
                    depth_m = load_depth_m(depth_path, (h, w), params.depth_scale)
                except Exception as exc:
                    missing_depth_images += 1
                    print(f"[WARN] failed to load depth for {image_path.name}: {exc}")
            refined, stats = apply_depth_refine(preds, depth_m, params)
            depth_preds.extend(refined)
            merge_stats(depth_stats, stats)

        if (image_id + 1) % 50 == 0 or image_id + 1 == len(images):
            print(f"Processed {image_id + 1}/{len(images)} images")

    result: Dict[str, object] = {
        "images": len(images),
        "gt_instances": len(all_gts),
        "classes": names,
    }
    if args.mode in ("rgb", "both"):
        result["rgb"] = evaluate_map(rgb_preds, all_gts, class_ids, thresholds)
        result["rgb"]["pred_instances"] = len(rgb_preds)
    if args.mode in ("depth", "both"):
        result["depth_refine"] = evaluate_map(depth_preds, all_gts, class_ids, thresholds)
        result["depth_refine"]["pred_instances"] = len(depth_preds)
        result["depth_refine"]["depth_stats"] = depth_stats
        result["depth_refine"]["missing_depth_images"] = missing_depth_images

    print("\n=== Mask AP Summary ===")
    if "rgb" in result:
        rgb = result["rgb"]
        print(
            "RGB-only       "
            f"mAP50={rgb['mask_mAP50']:.4f} "
            f"mAP50-95={rgb['mask_mAP50_95']:.4f} "
            f"pred={rgb['pred_instances']}"
        )
    if "depth_refine" in result:
        depth = result["depth_refine"]
        print(
            "RGB+DepthRefine "
            f"mAP50={depth['mask_mAP50']:.4f} "
            f"mAP50-95={depth['mask_mAP50_95']:.4f} "
            f"pred={depth['pred_instances']} "
            f"missing_depth_images={depth['missing_depth_images']}"
        )
        print(f"Depth refine stats: {depth['depth_stats']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
