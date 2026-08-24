# -*- coding: utf-8 -*-
"""
background_target_augmentation_v2.py

YOLO-seg 데이터셋용 배경 타겟 증강 스크립트 v2.

v1 대비 핵심 수정
1) hard_bg에서 object 주변에 원본 배경 띠(halo)가 생기던 문제 제거
   - object 주변을 dilate해서 원본으로 보존하지 않음
   - 대신 object mask 경계만 약하게 feathering
2) clutter 기본 제외
   - all 모드는 bg_color + hard_bg만 사용
3) hard_bg 강도 완화
   - 배경을 object 색으로 과도하게 끌고 가지 않음
   - 원본 배경 texture/명도를 더 많이 유지
4) label geometry는 바꾸지 않으므로 YOLO-seg label은 그대로 복사

입력 구조
INPUT_ROOT/
 ├─ images/
 └─ labels/

출력 구조
OUTPUT_ROOT/
 ├─ images/
 └─ labels/

실행 예시
python background_target_augmentation_v2.py --mode hard_bg --num-aug 2 --limit 20 --overwrite
python background_target_augmentation_v2.py --mode bg_color --num-aug 2 --limit 20 --overwrite
python background_target_augmentation_v2.py --mode all --num-aug 3 --overwrite
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ============================================================
# 기본 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = PROJECT_ROOT / "data" / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "dataset_aug_v1"


# ============================================================
# 기본 설정
# ============================================================

IMAGE_DIR_NAME = "images"
LABEL_DIR_NAME = "labels"

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

COPY_ORIGINALS = True
SAVE_EXT = ".jpg"
JPEG_QUALITY = 95

# v2에서는 halo 방지를 위해 dilate 보호 띠를 기본적으로 사용하지 않음.
# object 내부 보존은 mask 자체로만 수행.
EDGE_PROTECT_DILATE = 0

# 경계는 아주 약하게만 feather.
# 너무 크면 label 경계와 이미지 경계가 흐려질 수 있음.
FEATHER_KERNEL = 3

RANDOM_SEED: Optional[int] = 42


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    stem: str


# ============================================================
# 유틸
# ============================================================

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(output_root: Path) -> Tuple[Path, Path]:
    out_img_dir = output_root / IMAGE_DIR_NAME
    out_lbl_dir = output_root / LABEL_DIR_NAME
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    return out_img_dir, out_lbl_dir


def find_images(image_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(image_dir.glob(f"*{ext}"))
        paths.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def collect_samples(input_root: Path) -> List[Sample]:
    image_dir = input_root / IMAGE_DIR_NAME
    label_dir = input_root / LABEL_DIR_NAME

    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    samples: List[Sample] = []
    for img_path in find_images(image_dir):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] label missing, skip: {img_path.name}")
            continue
        samples.append(Sample(img_path, label_path, img_path.stem))

    if len(samples) == 0:
        raise RuntimeError(f"No valid image-label pairs found in {input_root}")

    return samples


def imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def imwrite_bgr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    elif ext == ".png":
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(ext, img)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    buf.tofile(str(path))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ============================================================
# YOLO-seg label -> mask
# ============================================================

def parse_yolo_seg_label(label_path: Path, w: int, h: int) -> List[Tuple[int, np.ndarray]]:
    objects: List[Tuple[int, np.ndarray]] = []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return objects

    for line_idx, line in enumerate(text.splitlines()):
        parts = line.strip().split()
        if len(parts) < 7:
            print(f"[WARN] invalid label line skip: {label_path.name}:{line_idx + 1}")
            continue

        try:
            cls_id = int(float(parts[0]))
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            print(f"[WARN] parse fail label line skip: {label_path.name}:{line_idx + 1}")
            continue

        if len(coords) % 2 != 0:
            coords = coords[:-1]

        pts = coords.reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * w
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * h

        poly = np.round(pts).astype(np.int32)
        if poly.shape[0] >= 3:
            objects.append((cls_id, poly))

    return objects


def build_object_mask(label_path: Path, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    objects = parse_yolo_seg_label(label_path, w, h)

    for _, poly in objects:
        cv2.fillPoly(mask, [poly], 255)

    return mask


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def make_alpha_from_mask(mask: np.ndarray, feather_kernel: int = FEATHER_KERNEL) -> np.ndarray:
    alpha = mask.astype(np.float32) / 255.0

    if feather_kernel and feather_kernel > 1:
        k = feather_kernel
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    return alpha[..., None]


def composite_keep_object(original: np.ndarray, aug_background: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """
    object mask 내부는 원본, mask 바깥은 증강 배경.
    v2에서는 dilated original ring을 만들지 않는다.
    """
    preserve_mask = dilate_mask(object_mask, EDGE_PROTECT_DILATE)
    alpha = make_alpha_from_mask(preserve_mask)
    out = original.astype(np.float32) * alpha + aug_background.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def get_mean_color(img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    idx = mask > 0
    if idx.sum() < 10:
        return None
    return img[idx].astype(np.float32).mean(axis=0)


def get_background_mask(object_mask: np.ndarray) -> np.ndarray:
    return (object_mask == 0)


def get_background_mean_color(img: np.ndarray, object_mask: np.ndarray) -> Optional[np.ndarray]:
    idx = get_background_mask(object_mask)
    if idx.sum() < 10:
        return None
    return img[idx].astype(np.float32).mean(axis=0)


# ============================================================
# 증강 1: Background-only color augmentation
# ============================================================

def adjust_hsv_background_candidate(img: np.ndarray) -> np.ndarray:
    """
    전체 이미지에 약한 색 변환 후보를 만들고,
    최종 합성에서 background 영역만 사용.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue_shift = random.uniform(-5, 5)
    sat_scale = random.uniform(0.85, 1.15)
    val_scale = random.uniform(0.85, 1.15)
    val_bias = random.uniform(-10, 10)

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale + val_bias, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = random.uniform(0.92, 1.08)
    beta = random.uniform(-6, 6)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        tint = np.array([
            random.uniform(-6, 6),
            random.uniform(-6, 6),
            random.uniform(-6, 6),
        ], dtype=np.float32)
        out = np.clip(out.astype(np.float32) + tint, 0, 255).astype(np.uint8)

    return out


def augment_bg_color(img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    aug_bg = adjust_hsv_background_candidate(img)
    return composite_keep_object(img, aug_bg, object_mask)


# ============================================================
# 증강 2: Hard background, but no halo
# ============================================================

def create_soft_low_contrast_background(img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """
    object 평균색 쪽으로 배경을 약하게 이동시켜 low-contrast hard case를 만든다.
    v1과 달리:
    - 내부에서 object를 한 번 합성하지 않음
    - 원본 배경 texture를 많이 보존
    - target 색을 object 평균색에 과도하게 고정하지 않음
    """
    obj_mean = get_mean_color(img, object_mask)
    bg_mean = get_background_mean_color(img, object_mask)

    if obj_mean is None or bg_mean is None:
        return adjust_hsv_background_candidate(img)

    # object 평균색으로 바로 가면 배경이 너무 비현실적으로 바뀌므로
    # background mean -> object mean 방향으로 일부만 이동.
    pull = random.uniform(0.25, 0.50)
    target_color = bg_mean * (1.0 - pull) + obj_mean * pull

    # 약간의 색 흔들림
    offset = np.array([
        random.uniform(-8, 8),
        random.uniform(-8, 8),
        random.uniform(-8, 8),
    ], dtype=np.float32)
    target_color = np.clip(target_color + offset, 0, 255)

    # 원본 배경의 명암/texture 보존
    img_f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=random.uniform(3, 8))
    local_texture = gray - gray_blur
    local_texture = np.clip(local_texture, -25, 25)

    target = np.ones_like(img_f) * target_color.reshape(1, 1, 3)
    target = target + local_texture[..., None] * random.uniform(0.35, 0.75)
    target = np.clip(target, 0, 255)

    # 핵심: 원본 배경과 target을 약하게 섞는다.
    mix = random.uniform(0.25, 0.55)
    aug_bg = img_f * (1.0 - mix) + target * mix

    # 너무 전체가 탁해지는 것을 방지하는 약한 gamma/contrast 보정
    if random.random() < 0.5:
        contrast = random.uniform(0.96, 1.04)
        aug_bg = (aug_bg - 127.5) * contrast + 127.5

    return np.clip(aug_bg, 0, 255).astype(np.uint8)


def augment_hard_bg(img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    aug_bg = create_soft_low_contrast_background(img, object_mask)
    return composite_keep_object(img, aug_bg, object_mask)


# ============================================================
# 증강 3: safe all
# ============================================================

def augment_all(img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """
    v2의 all은 clutter를 제외.
    지금 문제는 복잡한 박스/패치가 아니라 배경색/저대비 robustness가 핵심이므로
    bg_color + hard_bg만 섞는다.
    """
    if random.random() < 0.45:
        return augment_bg_color(img, object_mask)
    return augment_hard_bg(img, object_mask)


def apply_augmentation(mode: str, img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    if mode == "bg_color":
        return augment_bg_color(img, object_mask)
    if mode == "hard_bg":
        return augment_hard_bg(img, object_mask)
    if mode == "all":
        return augment_all(img, object_mask)

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# 저장
# ============================================================

def save_original(sample: Sample, img: np.ndarray, out_img_dir: Path, out_lbl_dir: Path) -> None:
    out_img_path = out_img_dir / f"{sample.stem}{SAVE_EXT}"
    out_lbl_path = out_lbl_dir / f"{sample.stem}.txt"

    imwrite_bgr(out_img_path, img)
    copy_file(sample.label_path, out_lbl_path)


def save_augmented(
    sample: Sample,
    aug_img: np.ndarray,
    out_img_dir: Path,
    out_lbl_dir: Path,
    mode: str,
    aug_idx: int,
) -> None:
    aug_stem = f"{sample.stem}_aug_{mode}_{aug_idx:02d}"
    out_img_path = out_img_dir / f"{aug_stem}{SAVE_EXT}"
    out_lbl_path = out_lbl_dir / f"{aug_stem}.txt"

    imwrite_bgr(out_img_path, aug_img)
    copy_file(sample.label_path, out_lbl_path)


def process_dataset(
    input_root: Path,
    output_root: Path,
    mode: str,
    num_aug: int,
    copy_originals: bool,
    overwrite: bool,
    limit: Optional[int] = None,
) -> None:
    if mode not in ["bg_color", "hard_bg", "all"]:
        raise ValueError("--mode must be one of: bg_color, hard_bg, all")

    if overwrite and output_root.exists():
        print(f"[INFO] remove existing output: {output_root}")
        shutil.rmtree(output_root)

    out_img_dir, out_lbl_dir = ensure_dirs(output_root)
    samples = collect_samples(input_root)

    if limit is not None and limit > 0:
        samples = samples[:limit]

    print("=" * 70)
    print("Background Target Augmentation v2")
    print("=" * 70)
    print(f"input_root    : {input_root}")
    print(f"output_root   : {output_root}")
    print(f"mode          : {mode}")
    print(f"num_aug/img   : {num_aug}")
    print(f"copy_originals: {copy_originals}")
    print(f"num_samples   : {len(samples)}")
    print("=" * 70)

    total_original = 0
    total_aug = 0
    skipped_empty_mask = 0

    for idx, sample in enumerate(samples, start=1):
        img = imread_bgr(sample.image_path)
        h, w = img.shape[:2]
        object_mask = build_object_mask(sample.label_path, w, h)

        if copy_originals:
            save_original(sample, img, out_img_dir, out_lbl_dir)
            total_original += 1

        if (object_mask > 0).sum() < 10:
            skipped_empty_mask += 1
            print(f"[WARN] empty object mask, only original copied: {sample.image_path.name}")
            continue

        for aug_idx in range(1, num_aug + 1):
            aug_img = apply_augmentation(mode, img, object_mask)
            save_augmented(sample, aug_img, out_img_dir, out_lbl_dir, mode, aug_idx)
            total_aug += 1

        if idx % 50 == 0 or idx == len(samples):
            print(f"[{idx:5d}/{len(samples):5d}] originals={total_original}, augmented={total_aug}")

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"saved originals     : {total_original}")
    print(f"saved augmented     : {total_aug}")
    print(f"empty mask skipped  : {skipped_empty_mask}")
    print(f"output images       : {out_img_dir}")
    print(f"output labels       : {out_lbl_dir}")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO-seg background-targeted augmentation script v2"
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help="입력 dataset root (images/, labels/).",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="증강 dataset 출력 root.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["bg_color", "hard_bg", "all"],
        help="증강 모드 선택: bg_color, hard_bg, all",
    )

    parser.add_argument(
        "--num-aug",
        type=int,
        default=3,
        help="원본 이미지 1장당 생성할 증강 이미지 개수",
    )

    parser.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="원본 이미지를 output에 복사하지 않음",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 output 폴더가 있으면 삭제 후 새로 생성",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="디버깅용. 앞에서 N개 샘플만 처리. 전체 처리하려면 생략.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="random seed. 매번 다르게 하려면 -1 입력.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = None if args.seed == -1 else args.seed
    set_seed(seed)

    process_dataset(
        input_root=args.input_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        mode=args.mode,
        num_aug=args.num_aug,
        copy_originals=not args.no_copy_originals,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
