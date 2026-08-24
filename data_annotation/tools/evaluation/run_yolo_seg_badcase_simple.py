#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_yolo_seg_badcase_simple.py

설명:
    YOLO-seg 모델을 매번 val 수행한 뒤, 생성된 predictions.json을 읽어
    segmentation 품질이 안 좋은 샘플을 "문제 유형별 디렉토리"로 저장한다.

변경 사항 (rev):
    - 파일명 유지: run_yolo_seg_badcase_simple.py
    - overlay 단순화:
        * GT              : 초록 contour
        * 정상/같은 class pred : 하늘색(cyan) contour
        * 틀린 pred / FP   : 빨간 contour
      -> fill 없이 contour 위주로만 그려서 가시성 개선
    - 텍스트 박스 축소
    - class_mismatch 판정 더 엄격하게 수정:
        * 다른 class pred가 GT와 충분히 겹치고
        * 같은 class pred는 miss 수준일 때만 mismatch로 판정
      -> 과도한 mismatch 감소 목적
    - issue별 디렉토리 유지:
        bad_cases/
            missed_detection/
            class_mismatch/
            low_mask_iou/
            low_confidence/
            false_positive/
            worst_all/

사용 예시:
    python run_yolo_seg_badcase_simple.py \
        --weights /path/to/model.pt \
        --data /path/to/dataset.yaml \
        --output-dir results/generated/bad_cases \
        --imgsz 640 \
        --batch 16 \
        --device 0 \
        --workers 8 \
        --top-k 50
"""

import csv
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import yaml
import numpy as np


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# Utils
# ============================================================

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(base: Path, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def load_dataset_paths(data_yaml: str):
    data_yaml = Path(data_yaml).resolve()

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    else:
        root = root.resolve()

    if "val" not in data:
        raise ValueError("dataset.yaml에 val 항목이 없습니다.")

    val_img_dir = resolve_path(root, data["val"]).resolve()
    label_dir = infer_label_dir(val_img_dir, root)

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        names = {}

    return {
        "root": root,
        "val_img_dir": val_img_dir,
        "label_dir": label_dir,
        "names": names,
    }


def infer_label_dir(img_dir: Path, root: Path) -> Path:
    candidates = []
    parts = list(img_dir.parts)

    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        new_parts = parts.copy()
        new_parts[idx] = "labels"
        candidates.append(Path(*new_parts))

    if img_dir.name == "images":
        candidates.append(img_dir.parent / "labels")

    candidates.append(root / "labels")
    candidates.append(root / "labels" / img_dir.name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    return candidates[0].resolve()


def collect_images(img_dir: Path):
    if not img_dir.exists():
        raise FileNotFoundError(f"val image dir가 없습니다: {img_dir}")

    paths = []
    for ext in IMG_EXTS:
        paths.extend(img_dir.rglob(f"*{ext}"))
        paths.extend(img_dir.rglob(f"*{ext.upper()}"))

    paths = sorted(set(paths))
    if len(paths) == 0:
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {img_dir}")

    return paths


def image_key(path: Path):
    return path.stem


def cls_name(cls, names):
    if cls in names:
        return str(names[cls])
    return f"class_{cls}"


# ============================================================
# Mask conversion
# ============================================================

def yolo_poly_to_mask(poly_norm, h, w):
    if len(poly_norm) < 6:
        return np.zeros((h, w), dtype=np.uint8)

    pts = np.array(poly_norm, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = np.round(pts).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def coco_seg_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    if segmentation is None:
        return mask

    # Polygon list
    if isinstance(segmentation, list):
        for poly in segmentation:
            if poly is None:
                continue

            if isinstance(poly, dict):
                m = coco_seg_to_mask(poly, h, w)
                mask = np.logical_or(mask, m).astype(np.uint8)
                continue

            if len(poly) < 6:
                continue

            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

        return mask

    # RLE dict
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils

            rle = segmentation
            if isinstance(rle.get("counts", None), list):
                rle = mask_utils.frPyObjects(rle, h, w)

            decoded = mask_utils.decode(rle)

            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)

            decoded = decoded.astype(np.uint8)

            if decoded.shape[:2] != (h, w):
                decoded = cv2.resize(decoded, (w, h), interpolation=cv2.INTER_NEAREST)

            return decoded.astype(np.uint8)

        except Exception:
            return mask

    return mask


def compute_iou(m1, m2):
    m1 = m1 > 0
    m2 = m2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def image_map5095_like(gt_ious, num_gt):
    if num_gt == 0:
        return 1.0

    ths = np.arange(0.50, 1.00, 0.05)
    return float(np.mean([sum(i >= th for i in gt_ious) / num_gt for th in ths]))


# ============================================================
# YOLO val
# ============================================================

def run_yolo_val(args, val_project_dir: Path):
    if val_project_dir.exists() and args.overwrite_val:
        shutil.rmtree(val_project_dir)

    safe_mkdir(val_project_dir)

    cmd = [
        "yolo",
        "segment",
        "val",
        f"model={args.weights}",
        f"data={args.data}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"conf={args.conf}",
        f"iou={args.iou}",
        "save_json=True",
        "plots=True",
        f"project={str(val_project_dir)}",
        f"name={args.run_name}",
        "exist_ok=True",
    ]

    if args.half:
        cmd.append("half=True")

    print("\n[1/6] Running YOLO segment val...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    run_dir = val_project_dir / args.run_name
    pred_json = run_dir / "predictions.json"

    if not pred_json.exists():
        raise FileNotFoundError(f"predictions.json을 찾지 못했습니다: {pred_json}")

    print(f"YOLO val result dir: {run_dir}")
    print(f"Pred JSON          : {pred_json}")

    return run_dir, pred_json


# ============================================================
# Load GT / prediction
# ============================================================

def load_gt(img_paths, label_dir: Path):
    print("\n[2/6] Loading GT masks...")

    gt_by_img = defaultdict(list)
    class_counter = Counter()
    missing = 0
    total = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        key = image_key(img_path)
        label_path = label_dir / f"{key}.txt"

        if not label_path.exists():
            missing += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                cls = int(float(parts[0]))
                poly = list(map(float, parts[1:]))
                mask = yolo_poly_to_mask(poly, h, w)

                if mask.sum() == 0:
                    continue

                gt_by_img[key].append({
                    "cls": cls,
                    "mask": mask,
                })
                class_counter[cls] += 1
                total += 1

    print(f"GT images loaded   : {len(gt_by_img)}")
    print(f"Total GT instances : {total}")
    print(f"Missing labels     : {missing}")
    print(f"GT class counts    : {dict(sorted(class_counter.items()))}")

    return gt_by_img


def load_raw_preds(pred_json: Path, img_paths):
    print("\n[3/6] Loading raw predictions...")

    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    stems = {p.stem for p in img_paths}
    names = {p.name: p.stem for p in img_paths}

    raw_by_img = defaultdict(list)
    raw_cls_counter = Counter()
    seg_type_counter = Counter()

    for p in preds:
        image_id = str(p.get("image_id", ""))

        if image_id in stems:
            key = image_id
        elif image_id in names:
            key = names[image_id]
        else:
            key = Path(image_id).stem

        raw_cls = int(p.get("category_id", 0))
        seg = p.get("segmentation", None)

        if isinstance(seg, list):
            seg_type_counter["polygon_list"] += 1
        elif isinstance(seg, dict):
            seg_type_counter["rle_dict"] += 1
        elif seg is None:
            seg_type_counter["none"] += 1
        else:
            seg_type_counter[type(seg).__name__] += 1

        raw_by_img[key].append({
            "raw_cls": raw_cls,
            "score": float(p.get("score", 0.0)),
            "segmentation": seg,
        })

        raw_cls_counter[raw_cls] += 1

    print(f"Pred images loaded : {len(raw_by_img)}")
    print(f"Total predictions  : {sum(len(v) for v in raw_by_img.values())}")
    print(f"Raw pred class cnt : {dict(sorted(raw_cls_counter.items()))}")
    print(f"Segmentation types : {dict(seg_type_counter)}")

    return raw_by_img


def choose_category_offset(gt_by_img, raw_pred_by_img, output_dir: Path):
    print("\n[4/6] Choosing category_id offset...")

    scores = {}

    for offset in [0, 1]:
        hit = 0
        total = 0
        img_hit = 0

        for key, preds in raw_pred_by_img.items():
            gt_classes = {g["cls"] for g in gt_by_img.get(key, [])}
            if len(gt_classes) == 0:
                continue

            pred_classes = [p["raw_cls"] - offset for p in preds if p["raw_cls"] - offset >= 0]
            total += len(pred_classes)

            local_hit = sum(c in gt_classes for c in pred_classes)
            hit += local_hit

            if local_hit > 0:
                img_hit += 1

        ratio = hit / max(total, 1)
        scores[offset] = (ratio, hit, total, img_hit)

    chosen = max(scores.keys(), key=lambda k: (scores[k][0], scores[k][3]))

    with open(output_dir / "category_offset.txt", "w", encoding="utf-8") as f:
        for offset, (ratio, hit, total, img_hit) in scores.items():
            f.write(f"offset={offset}, ratio={ratio:.6f}, hits={hit}/{total}, image_hits={img_hit}\n")
        f.write(f"chosen_offset={chosen}\n")

    for offset, (ratio, hit, total, img_hit) in scores.items():
        print(f"offset={offset}: ratio={ratio:.4f}, hits={hit}/{total}, image_hits={img_hit}")
    print(f"Chosen offset: {chosen}")

    return chosen


def apply_category_offset(raw_pred_by_img, offset):
    pred_by_img = defaultdict(list)

    for key, preds in raw_pred_by_img.items():
        for p in preds:
            cls = p["raw_cls"] - offset
            if cls < 0:
                cls = -999999

            pred_by_img[key].append({
                "cls": cls,
                "raw_cls": p["raw_cls"],
                "score": p["score"],
                "segmentation": p["segmentation"],
            })

    return pred_by_img


# ============================================================
# Analyze bad cases
# ============================================================

def analyze_one_image(img_path, gts, preds, names, args):
    """
    우선순위:
        1) class_mismatch
        2) missed_detection
        3) low_mask_iou
        4) low_confidence
        5) false_positive
        6) good

    class_mismatch를 더 엄격하게:
        - 다른 class pred가 GT와 IoU >= class_mismatch_iou
        - 같은 class pred는 miss 수준 (same_iou < miss_iou)
        - 다른 class pred score도 일정 이상
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    pred_items = []
    for p in preds:
        m = coco_seg_to_mask(p["segmentation"], h, w)
        q = dict(p)
        q["mask"] = m
        q["area"] = int(m.sum())
        pred_items.append(q)

    gt_records = []

    for gi, gt in enumerate(gts):
        same_best = None
        any_best = None

        for pi, p in enumerate(pred_items):
            iou = compute_iou(gt["mask"], p["mask"])

            cand = {
                "gt_idx": gi,
                "pred_idx": pi,
                "gt_cls": gt["cls"],
                "pred_cls": p["cls"],
                "score": p["score"],
                "iou": iou,
            }

            if any_best is None or iou > any_best["iou"]:
                any_best = cand

            if p["cls"] == gt["cls"]:
                if same_best is None or iou > same_best["iou"]:
                    same_best = cand

        same_iou = same_best["iou"] if same_best is not None else 0.0
        any_iou = any_best["iou"] if any_best is not None else 0.0
        same_score = same_best["score"] if same_best is not None else 0.0
        any_score = any_best["score"] if any_best is not None else 0.0
        any_pred_cls = any_best["pred_cls"] if any_best is not None else None

        issue = "good"

        # 더 엄격한 class mismatch 판정
        if (
            any_best is not None
            and any_pred_cls != gt["cls"]
            and any_iou >= args.class_mismatch_iou
            and any_score >= args.class_mismatch_conf
            and same_iou < args.miss_iou
        ):
            issue = "class_mismatch"

        elif same_iou < args.miss_iou:
            issue = "missed_detection"

        elif same_iou < args.bad_iou:
            issue = "low_mask_iou"

        elif same_score < args.low_conf:
            issue = "low_confidence"

        gt_records.append({
            "gt_idx": gi,
            "gt_cls": gt["cls"],
            "gt_cls_name": cls_name(gt["cls"], names),
            "same_iou": same_iou,
            "same_score": same_score,
            "same_pred_cls": same_best["pred_cls"] if same_best is not None else None,
            "any_iou": any_iou,
            "any_score": any_score,
            "any_pred_cls": any_pred_cls,
            "any_pred_cls_name": cls_name(any_pred_cls, names) if any_best is not None else "none",
            "issue": issue,
            "same_best": same_best,
            "any_best": any_best,
        })

    # FP: GT와 충분히 겹치지 않는 예측
    fp_preds = []
    for pi, p in enumerate(pred_items):
        if p["score"] < args.fp_conf:
            continue

        best_any_iou = 0.0
        for gt in gts:
            best_any_iou = max(best_any_iou, compute_iou(gt["mask"], p["mask"]))

        if best_any_iou < args.fp_iou:
            fp_preds.append({
                "pred_idx": pi,
                "pred_cls": p["cls"],
                "pred_cls_name": cls_name(p["cls"], names),
                "score": p["score"],
                "best_iou": best_any_iou,
            })

    issue_priority = [
        "class_mismatch",
        "missed_detection",
        "low_mask_iou",
        "low_confidence",
    ]

    issue_counts = Counter([r["issue"] for r in gt_records])
    fp_count = len(fp_preds)

    primary_issue = "good"
    for issue in issue_priority:
        if issue_counts[issue] > 0:
            primary_issue = issue
            break

    if primary_issue == "good" and fp_count > 0:
        primary_issue = "false_positive"

    gt_ious = [r["same_iou"] for r in gt_records]
    map_like = image_map5095_like(gt_ious, len(gts))

    avg_iou = float(np.mean(gt_ious)) if len(gt_ious) else 0.0
    min_iou = float(np.min(gt_ious)) if len(gt_ious) else 0.0

    quality_score = (
        map_like
        - args.miss_penalty * (issue_counts["missed_detection"] / max(len(gts), 1))
        - args.class_penalty * (issue_counts["class_mismatch"] / max(len(gts), 1))
        - args.bad_penalty * (issue_counts["low_mask_iou"] / max(len(gts), 1))
        - args.low_conf_penalty * (issue_counts["low_confidence"] / max(len(gts), 1))
        - args.fp_penalty * (fp_count / max(len(pred_items), 1))
    )

    return {
        "image": str(img_path),
        "key": image_key(Path(img_path)),
        "primary_issue": primary_issue,
        "quality_score": float(quality_score),
        "map5095_like": float(map_like),
        "avg_iou": avg_iou,
        "min_iou": min_iou,
        "gt_count": len(gts),
        "pred_count": len(pred_items),
        "missed_detection_count": issue_counts["missed_detection"],
        "class_mismatch_count": issue_counts["class_mismatch"],
        "low_mask_iou_count": issue_counts["low_mask_iou"],
        "low_confidence_count": issue_counts["low_confidence"],
        "false_positive_count": fp_count,
        "gt_records": gt_records,
        "fp_preds": fp_preds,
    }


def analyze_all(img_paths, gt_by_img, pred_by_img, names, args, output_dir):
    print("\n[5/6] Analyzing bad cases...")

    results = []
    debug = Counter()
    iou_sum = 0.0
    iou_n = 0
    pred_mask_total = 0
    pred_mask_nonempty = 0

    for img_path in img_paths:
        key = image_key(img_path)
        gts = gt_by_img.get(key, [])
        preds = pred_by_img.get(key, [])

        if len(gts) == 0:
            continue

        result = analyze_one_image(img_path, gts, preds, names, args)
        if result is None:
            continue

        results.append(result)

        debug["images"] += 1
        debug["total_gt"] += len(gts)
        debug["total_pred"] += len(preds)
        debug[f"issue_{result['primary_issue']}"] += 1

        for r in result["gt_records"]:
            iou_sum += r["same_iou"]
            iou_n += 1

        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            for p in preds:
                m = coco_seg_to_mask(p["segmentation"], h, w)
                pred_mask_total += 1
                if m.sum() > 0:
                    pred_mask_nonempty += 1

    results.sort(key=lambda x: x["quality_score"])

    with open(output_dir / "debug_summary.txt", "w", encoding="utf-8") as f:
        for k, v in sorted(debug.items()):
            f.write(f"{k}: {v}\n")
        f.write(f"avg_same_class_iou: {iou_sum / max(iou_n, 1):.6f}\n")
        f.write(f"pred_mask_nonempty: {pred_mask_nonempty}/{pred_mask_total}\n")

    print("Debug summary")
    for k, v in sorted(debug.items()):
        print(f"{k}: {v}")
    print(f"avg same-class IoU : {iou_sum / max(iou_n, 1):.4f}")
    print(f"pred mask nonempty : {pred_mask_nonempty}/{pred_mask_total}")

    return results


# ============================================================
# Overlay
# ============================================================

def draw_mask_contour(img, mask, color, thickness=2):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def draw_label_panel(img, lines):
    """
    더 작은 텍스트 패널.
    """
    x, y = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thick = 1
    line_h = 22

    max_w = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        max_w = max(max_w, tw)

    box_w = max_w + 20
    box_h = line_h * len(lines) + 12

    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)

    yy = y + 18
    for line in lines:
        cv2.putText(img, line, (x + 10, yy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        yy += line_h

    return img


def make_overlay(result, gt_by_img, pred_by_img, names, args):
    """
    단순 contour 기반 시각화:
      - GT contour              : green
      - 같은 class best pred    : cyan
      - 틀린 pred / FP          : red
    """
    img_path = Path(result["image"])
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    key = result["key"]

    gts = gt_by_img.get(key, [])
    preds = pred_by_img.get(key, [])

    pred_items = []
    for p in preds:
        q = dict(p)
        q["mask"] = coco_seg_to_mask(p["segmentation"], h, w)
        pred_items.append(q)

    vis = img.copy()

    # 기본: GT는 모두 green contour
    for gt in gts:
        vis = draw_mask_contour(vis, gt["mask"], color=(0, 255, 0), thickness=2)

    issue = result["primary_issue"]

    detail_lines = []

    # 같은 class best / any best prediction만 relevant하게 그림
    if issue == "missed_detection":
        bad = [r for r in result["gt_records"] if r["issue"] == "missed_detection"]
        if bad:
            r = bad[0]
            if r["same_best"] is not None:
                pi = r["same_best"]["pred_idx"]
                vis = draw_mask_contour(vis, pred_items[pi]["mask"], color=(255, 255, 0), thickness=2)  # cyan
            detail_lines = [
                "BAD CASE: MISSED DETECTION",
                f"GT: {r['gt_cls_name']}",
                f"best same IoU: {r['same_iou']:.3f}",
            ]

    elif issue == "class_mismatch":
        bad = [r for r in result["gt_records"] if r["issue"] == "class_mismatch"]
        if bad:
            r = bad[0]
            if r["any_best"] is not None:
                pi = r["any_best"]["pred_idx"]
                vis = draw_mask_contour(vis, pred_items[pi]["mask"], color=(0, 0, 255), thickness=2)  # red
            detail_lines = [
                "BAD CASE: CLASS MISMATCH",
                f"GT: {r['gt_cls_name']}",
                f"Pred: {r['any_pred_cls_name']}",
                f"IoU: {r['any_iou']:.3f} conf: {r['any_score']:.3f}",
            ]

    elif issue == "low_mask_iou":
        bad = [r for r in result["gt_records"] if r["issue"] == "low_mask_iou"]
        if bad:
            r = bad[0]
            if r["same_best"] is not None:
                pi = r["same_best"]["pred_idx"]
                vis = draw_mask_contour(vis, pred_items[pi]["mask"], color=(255, 255, 0), thickness=2)  # cyan
            detail_lines = [
                "BAD CASE: LOW MASK IOU",
                f"Class: {r['gt_cls_name']}",
                f"IoU: {r['same_iou']:.3f}",
                f"conf: {r['same_score']:.3f}",
            ]

    elif issue == "low_confidence":
        bad = [r for r in result["gt_records"] if r["issue"] == "low_confidence"]
        if bad:
            r = bad[0]
            if r["same_best"] is not None:
                pi = r["same_best"]["pred_idx"]
                vis = draw_mask_contour(vis, pred_items[pi]["mask"], color=(255, 255, 0), thickness=2)  # cyan
            detail_lines = [
                "BAD CASE: LOW CONFIDENCE",
                f"Class: {r['gt_cls_name']}",
                f"conf: {r['same_score']:.3f}",
                f"IoU: {r['same_iou']:.3f}",
            ]

    elif issue == "false_positive":
        fp = result["fp_preds"][0] if result["fp_preds"] else None
        if fp:
            pi = fp["pred_idx"]
            vis = draw_mask_contour(vis, pred_items[pi]["mask"], color=(0, 0, 255), thickness=2)  # red
            detail_lines = [
                "BAD CASE: FALSE POSITIVE",
                f"Pred: {fp['pred_cls_name']}",
                f"conf: {fp['score']:.3f}",
                f"best GT IoU: {fp['best_iou']:.3f}",
            ]

    else:
        detail_lines = [
            "GOOD CASE",
            f"mAP-like: {result['map5095_like']:.3f}",
            f"avg IoU: {result['avg_iou']:.3f}",
        ]

    detail_lines.append(f"score: {result['quality_score']:.3f}")
    vis = draw_label_panel(vis, detail_lines)

    return vis


# ============================================================
# Save outputs
# ============================================================

def save_report(results, path: Path):
    fields = [
        "rank",
        "primary_issue",
        "image",
        "quality_score",
        "map5095_like",
        "avg_iou",
        "min_iou",
        "gt_count",
        "pred_count",
        "missed_detection_count",
        "class_mismatch_count",
        "low_mask_iou_count",
        "low_confidence_count",
        "false_positive_count",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rank, r in enumerate(results, 1):
            row = {k: r.get(k, "") for k in fields}
            row["rank"] = rank
            writer.writerow(row)


def save_outputs(results, gt_by_img, pred_by_img, names, output_dir: Path, args, val_run_dir: Path):
    print("\n[6/6] Saving simplified bad case overlays...")

    bad_root = output_dir / "bad_cases"
    worst_all_dir = bad_root / "worst_all"
    safe_mkdir(bad_root)
    safe_mkdir(worst_all_dir)

    issue_dirs = {
        "missed_detection": bad_root / "missed_detection",
        "class_mismatch": bad_root / "class_mismatch",
        "low_mask_iou": bad_root / "low_mask_iou",
        "low_confidence": bad_root / "low_confidence",
        "false_positive": bad_root / "false_positive",
    }

    for d in issue_dirs.values():
        safe_mkdir(d)

    save_report(results, output_dir / "report_all.csv")

    # 전체 worst top-k
    for rank, r in enumerate(results[:args.top_k], 1):
        if r["primary_issue"] == "good":
            continue

        vis = make_overlay(r, gt_by_img, pred_by_img, names, args)
        if vis is None:
            continue

        img_name = Path(r["image"]).name
        save_path = worst_all_dir / f"{rank:04d}_{r['primary_issue']}_{img_name}"
        cv2.imwrite(str(save_path), vis)

    # 문제 유형별 저장
    counts = Counter()
    for r in results:
        issue = r["primary_issue"]

        if issue == "good":
            continue

        if issue not in issue_dirs:
            continue

        if counts[issue] >= args.max_per_issue:
            continue

        vis = make_overlay(r, gt_by_img, pred_by_img, names, args)
        if vis is None:
            continue

        counts[issue] += 1
        img_name = Path(r["image"]).name
        save_path = issue_dirs[issue] / f"{counts[issue]:04d}_{img_name}"
        cv2.imwrite(str(save_path), vis)

    with open(output_dir / "val_run_dir.txt", "w", encoding="utf-8") as f:
        f.write(str(val_run_dir) + "\n")

    print("\nDone.")
    print(f"Output dir : {output_dir}")
    print(f"Report     : {output_dir / 'report_all.csv'}")
    print(f"Bad cases  : {bad_root}")

    print("\nSaved count by issue")
    for issue, count in counts.items():
        print(f"{issue}: {count}")

    print("\n===== WORST PREVIEW =====")
    shown = 0
    for i, r in enumerate(results, 1):
        if r["primary_issue"] == "good":
            continue
        print(
            f"{i:04d} | {r['primary_issue']:17s} | "
            f"q={r['quality_score']:.3f} | "
            f"map_like={r['map5095_like']:.3f} | "
            f"avgIoU={r['avg_iou']:.3f} | "
            f"{r['image']}"
        )
        shown += 1
        if shown >= min(args.top_k, 30):
            break


# ============================================================
# Args / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--half", action="store_true")

    p.add_argument("--val-project", type=str, default=None)
    p.add_argument("--run-name", type=str, default="val_badcase_simple")
    p.add_argument("--overwrite-val", action="store_true")

    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-per-issue", type=int, default=100)

    # Bad case thresholds
    p.add_argument("--miss-iou", type=float, default=0.50)
    p.add_argument("--class-mismatch-iou", type=float, default=0.50)
    p.add_argument("--class-mismatch-conf", type=float, default=0.25)
    p.add_argument("--bad-iou", type=float, default=0.70)
    p.add_argument("--low-conf", type=float, default=0.50)
    p.add_argument("--fp-iou", type=float, default=0.30)
    p.add_argument("--fp-conf", type=float, default=0.25)

    # Scoring penalties
    p.add_argument("--miss-penalty", type=float, default=0.50)
    p.add_argument("--class-penalty", type=float, default=0.40)
    p.add_argument("--bad-penalty", type=float, default=0.20)
    p.add_argument("--low-conf-penalty", type=float, default=0.10)
    p.add_argument("--fp-penalty", type=float, default=0.10)

    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    safe_mkdir(output_dir)

    val_project_dir = Path(args.val_project).resolve() if args.val_project else output_dir / "yolo_val"

    # 1. run val
    val_run_dir, pred_json = run_yolo_val(args, val_project_dir)

    # 2. dataset paths
    paths = load_dataset_paths(args.data)
    img_paths = collect_images(paths["val_img_dir"])

    print("\nDataset paths")
    print(f"root      : {paths['root']}")
    print(f"val images: {paths['val_img_dir']}")
    print(f"labels    : {paths['label_dir']}")
    print(f"num images: {len(img_paths)}")

    # 3. load GT / pred
    gt_by_img = load_gt(img_paths, paths["label_dir"])
    raw_pred_by_img = load_raw_preds(pred_json, img_paths)

    # 4. category offset
    offset = choose_category_offset(gt_by_img, raw_pred_by_img, output_dir)
    pred_by_img = apply_category_offset(raw_pred_by_img, offset)

    # 5. analyze
    results = analyze_all(img_paths, gt_by_img, pred_by_img, paths["names"], args, output_dir)

    if len(results) == 0:
        raise RuntimeError("분석 결과가 없습니다. label 경로 또는 predictions.json을 확인하세요.")

    # 6. save
    save_outputs(results, gt_by_img, pred_by_img, paths["names"], output_dir, args, val_run_dir)


if __name__ == "__main__":
    main()
