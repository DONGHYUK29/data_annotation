# -*- coding: utf-8 -*-
"""
light_aug_split_v1.py

YOLO-seg 데이터셋용 train/val split + 밝기/대비/CLAHE 증강 스크립트.

목적
1) 기존 flat 구조 dataset/images, dataset/labels 를 8:2 train/val 로 분리
2) 배경 종류 + class 기준으로 stratified split
   예: desk_1_3.png
       - background = desk
       - class      = 1
       - index      = 3
3) train에만 밝기 증강 이미지 추가
4) val은 원본 이미지만 유지
5) label geometry는 바뀌지 않으므로 YOLO-seg label은 그대로 복사

입력 구조
INPUT_ROOT/
 ├─ images/
 │   ├─ desk_1_3.png
 │   └─ ...
 └─ labels/
     ├─ desk_1_3.txt
     └─ ...

출력 구조
OUTPUT_ROOT/
 ├─ images/
 │   ├─ train/
 │   └─ val/
 ├─ labels/
 │   ├─ train/
 │   └─ val/
 └─ split/
     ├─ train.txt
     └─ val.txt

실행 예시
python create/light_aug_split_v1.py --overwrite --num-aug 2 --mode all
python create/light_aug_split_v1.py --overwrite --num-aug 1 --mode clahe
python create/light_aug_split_v1.py --overwrite --limit 30 --num-aug 2 --mode all
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 기본 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = PROJECT_ROOT / "data" / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "dataset_light_aug_v1"

IMAGE_DIR_NAME = "images"
LABEL_DIR_NAME = "labels"

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

TRAIN_RATIO = 0.8
RANDOM_SEED: Optional[int] = 42

# 원본 이미지는 그대로 copy하고,
# 증강 이미지만 아래 확장자로 저장.
AUG_SAVE_EXT = ".jpg"
JPEG_QUALITY = 95


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    stem: str
    background: str
    cls_name: str
    index_name: str


# ============================================================
# 유틸
# ============================================================

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def find_images(image_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(image_dir.glob(f"*{ext}"))
        paths.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def parse_stem(stem: str) -> Tuple[str, str, str]:
    """
    예:
      desk_1_3     -> background=desk, cls=1, index=3
      floor_10_66  -> background=floor, cls=10, index=66

    혹시 background 이름에 underscore가 들어가도 뒤에서 2개만 숫자로 봄.
    예:
      white_desk_1_3 -> background=white_desk, cls=1, index=3
    """
    parts = stem.split("_")
    if len(parts) < 3:
        # fallback
        return "unknown_bg", "unknown_cls", stem

    background = "_".join(parts[:-2])
    cls_name = parts[-2]
    index_name = parts[-1]
    return background, cls_name, index_name


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

        bg, cls_name, index_name = parse_stem(img_path.stem)

        samples.append(
            Sample(
                image_path=img_path,
                label_path=label_path,
                stem=img_path.stem,
                background=bg,
                cls_name=cls_name,
                index_name=index_name,
            )
        )

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
        ok, buf = cv2.imencode(
            ext,
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
    elif ext == ".png":
        ok, buf = cv2.imencode(
            ext,
            img,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
    else:
        ok, buf = cv2.imencode(ext, img)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    buf.tofile(str(path))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    dirs = {
        "img_train": output_root / IMAGE_DIR_NAME / "train",
        "img_val": output_root / IMAGE_DIR_NAME / "val",
        "lbl_train": output_root / LABEL_DIR_NAME / "train",
        "lbl_val": output_root / LABEL_DIR_NAME / "val",
        "split": output_root / "split",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    return dirs


# ============================================================
# Split
# ============================================================

def stratified_split(
    samples: List[Sample],
    train_ratio: float,
    seed: Optional[int],
) -> Tuple[List[Sample], List[Sample]]:
    """
    background + class 기준으로 그룹을 나눠서 8:2 split.
    각 배경-class 조합의 분포가 train/val에 유지되도록 함.
    """
    rng = random.Random(seed)

    groups: Dict[Tuple[str, str], List[Sample]] = {}

    for s in samples:
        key = (s.background, s.cls_name)
        groups.setdefault(key, []).append(s)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    print("=" * 70)
    print("Split groups")
    print("=" * 70)

    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)

        n = len(group)
        n_val = int(round(n * (1.0 - train_ratio)))

        # 그룹에 샘플이 충분히 있으면 val 최소 1개 보장
        if n >= 2:
            n_val = max(1, n_val)

        # 전부 val로 빠지는 것 방지
        n_val = min(n_val, n - 1) if n >= 2 else 0

        val_part = group[:n_val]
        train_part = group[n_val:]

        train_samples.extend(train_part)
        val_samples.extend(val_part)

        print(
            f"group={key[0]} / class={key[1]:>2s} | "
            f"total={n:4d}, train={len(train_part):4d}, val={len(val_part):4d}"
        )

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    print("=" * 70)
    print(f"total train: {len(train_samples)}")
    print(f"total val  : {len(val_samples)}")
    print("=" * 70)

    return train_samples, val_samples


# ============================================================
# 밝기 계열 증강
# ============================================================

def clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(
    img: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    brightness: -255 ~ 255 근처
    contrast  : 1.0 기준
    """
    out = img.astype(np.float32)
    out = (out - 127.5) * contrast + 127.5 + brightness
    return clip_uint8(out)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    gamma > 1: 어두워짐
    gamma < 1: 밝아짐
    """
    gamma = max(gamma, 1e-6)

    table = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
        dtype=np.float32,
    )
    table = clip_uint8(table)
    return cv2.LUT(img, table)


def apply_clahe_lab(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    blend: float = 0.7,
) -> np.ndarray:
    """
    LAB 색공간의 L 채널에 CLAHE 적용.
    blend를 둬서 너무 인위적인 결과가 되지 않도록 함.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    out = img.astype(np.float32) * (1.0 - blend) + out.astype(np.float32) * blend
    return clip_uint8(out)


def apply_random_shadow(img: np.ndarray) -> np.ndarray:
    """
    실제 그림자 느낌을 약하게 추가.
    label geometry는 그대로 유지.
    과하게 쓰면 segmentation에 악영향이 있을 수 있으므로 약하게만 사용.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    mask = np.zeros((h, w), dtype=np.float32)

    # 랜덤 사각/타원 그림자
    if random.random() < 0.5:
        x1 = random.randint(0, max(0, w - 1))
        y1 = random.randint(0, max(0, h - 1))
        x2 = random.randint(0, max(0, w - 1))
        y2 = random.randint(0, max(0, h - 1))

        pts = np.array(
            [
                [x1, y1],
                [x2, y1 + random.randint(-h // 6, h // 6)],
                [x2 + random.randint(-w // 6, w // 6), y2],
                [x1 + random.randint(-w // 6, w // 6), y2],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 1.0)
    else:
        center = (
            random.randint(0, max(0, w - 1)),
            random.randint(0, max(0, h - 1)),
        )
        axes = (
            random.randint(max(10, w // 5), max(11, w // 2)),
            random.randint(max(10, h // 5), max(11, h // 2)),
        )
        angle = random.uniform(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)

    blur_k = random.choice([31, 41, 51, 61])
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = np.clip(mask, 0.0, 1.0)

    strength = random.uniform(0.08, 0.25)
    factor = 1.0 - mask[..., None] * strength

    out = out * factor
    return clip_uint8(out)


def apply_mild_highlight(img: np.ndarray) -> np.ndarray:
    """
    빛 반사/밝은 영역을 아주 약하게 추가.
    너무 강하면 흰색 patch 학습이 되어 악영향 가능성이 있으므로 낮은 확률로만 사용 권장.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    mask = np.zeros((h, w), dtype=np.float32)

    center = (
        random.randint(0, max(0, w - 1)),
        random.randint(0, max(0, h - 1)),
    )
    axes = (
        random.randint(max(10, w // 8), max(11, w // 3)),
        random.randint(max(10, h // 8), max(11, h // 3)),
    )
    angle = random.uniform(0, 180)

    cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)

    blur_k = random.choice([31, 41, 51])
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = np.clip(mask, 0.0, 1.0)

    strength = random.uniform(10, 35)
    out = out + mask[..., None] * strength

    return clip_uint8(out)

def recover_if_too_dark(
    img: np.ndarray,
    min_mean: float = 55.0,
    target_mean: float = 70.0,
) -> np.ndarray:
    """
    증강 후 이미지가 너무 어두워져서 정보가 사라지는 경우 자동 복구.
    grayscale 평균이 min_mean보다 낮으면 target_mean 근처로 살짝 올림.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())

    if mean_val >= min_mean:
        return img

    gain = target_mean / max(mean_val, 1.0)
    gain = min(gain, 1.45)  # 너무 과하게 밝히지 않음

    out = img.astype(np.float32) * gain
    return clip_uint8(out)

def augment_dark(img: np.ndarray) -> np.ndarray:
    """
    조도 낮은 상황 대응.
    너무 어둡게 죽이지 않고, 실제 시연에서 나올 법한 저조도 수준으로 제한.
    """
    out = img.copy()

    # 전체 밝기 감소: 기존보다 완화
    contrast = random.uniform(0.90, 1.12)
    brightness = random.uniform(-42, -8)
    out = adjust_brightness_contrast(out, brightness=brightness, contrast=contrast)

    # gamma도 기존보다 완화
    if random.random() < 0.65:
        gamma = random.uniform(1.08, 1.45)
        out = apply_gamma(out, gamma)

    # 너무 어두워져 정보가 죽으면 자동 복구
    out = recover_if_too_dark(out, min_mean=48.0, target_mean=63.0)

    # 일부 케이스는 저조도 + CLAHE로 윤곽 복구
    if random.random() < 0.35:
        out = apply_clahe_lab(
            out,
            clip_limit=random.uniform(1.3, 2.2),
            tile_grid_size=8,
            blend=random.uniform(0.20, 0.45),
        )

    return out


def augment_contrast(img: np.ndarray) -> np.ndarray:
    """
    일반적인 대비/밝기 흔들림.
    """
    brightness = random.uniform(-35, 25)
    contrast = random.uniform(0.65, 1.35)
    out = adjust_brightness_contrast(img, brightness=brightness, contrast=contrast)

    if random.random() < 0.5:
        gamma = random.uniform(0.75, 1.35)
        out = apply_gamma(out, gamma)

    return out


def augment_clahe(img: np.ndarray) -> np.ndarray:
    """
    어두운 환경에서 국소 대비 강화.
    너무 강하면 이미지가 부자연스러워져서 clipLimit/blend를 약하게 둠.
    """
    clip_limit = random.uniform(1.5, 3.0)
    tile_grid_size = random.choice([8, 8, 12])
    blend = random.uniform(0.45, 0.75)

    out = apply_clahe_lab(
        img,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        blend=blend,
    )

    # CLAHE 후 약간 어둡거나 밝은 상황도 섞음
    if random.random() < 0.5:
        brightness = random.uniform(-25, 10)
        contrast = random.uniform(0.90, 1.15)
        out = adjust_brightness_contrast(out, brightness=brightness, contrast=contrast)

    return out


def augment_shadow(img: np.ndarray) -> np.ndarray:
    """
    그림자/부분 조도 저하.
    너무 강한 검은 그림자 대신 실제 조도 변화에 가까운 약한 shadow.
    """
    out = img.copy()

    if random.random() < 0.5:
        out = adjust_brightness_contrast(
            out,
            brightness=random.uniform(-18, -3),
            contrast=random.uniform(0.92, 1.08),
        )

    out = apply_random_shadow(out)
    out = recover_if_too_dark(out, min_mean=55.0, target_mean=68.0)

    return out


def augment_mixed_light(img: np.ndarray) -> np.ndarray:
    """
    v1 기본 all 모드.
    너무 과격하지 않게 조명 변화 위주로 섞음.
    """
    r = random.random()

    if r < 0.35:
        out = augment_dark(img)
        mode_name = "dark"
    elif r < 0.60:
        out = augment_contrast(img)
        mode_name = "contrast"
    elif r < 0.80:
        out = augment_clahe(img)
        mode_name = "clahe"
    elif r < 0.95:
        out = augment_shadow(img)
        mode_name = "shadow"
    else:
        # 반사는 낮은 확률로만
        out = apply_mild_highlight(img)
        mode_name = "highlight"

    # 일부 케이스에 CLAHE 약하게 추가
    if mode_name != "clahe" and random.random() < 0.15:
        out = apply_clahe_lab(
            out,
            clip_limit=random.uniform(1.3, 2.2),
            tile_grid_size=8,
            blend=random.uniform(0.25, 0.45),
        )

    return out


def apply_augmentation(mode: str, img: np.ndarray) -> np.ndarray:
    if mode == "dark":
        return augment_dark(img)
    if mode == "contrast":
        return augment_contrast(img)
    if mode == "clahe":
        return augment_clahe(img)
    if mode == "shadow":
        return augment_shadow(img)
    if mode == "all":
        return augment_mixed_light(img)

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# 저장
# ============================================================

def copy_original_sample(
    sample: Sample,
    out_img_dir: Path,
    out_lbl_dir: Path,
) -> None:
    """
    원본 이미지는 재인코딩하지 않고 그대로 복사.
    """
    out_img_path = out_img_dir / sample.image_path.name
    out_lbl_path = out_lbl_dir / f"{sample.stem}.txt"

    copy_file(sample.image_path, out_img_path)
    copy_file(sample.label_path, out_lbl_path)


def save_augmented_sample(
    sample: Sample,
    aug_img: np.ndarray,
    out_img_dir: Path,
    out_lbl_dir: Path,
    mode: str,
    aug_idx: int,
    save_ext: str,
) -> None:
    aug_stem = f"{sample.stem}_aug_light_{mode}_{aug_idx:02d}"
    out_img_path = out_img_dir / f"{aug_stem}{save_ext}"
    out_lbl_path = out_lbl_dir / f"{aug_stem}.txt"

    imwrite_bgr(out_img_path, aug_img)
    copy_file(sample.label_path, out_lbl_path)


def write_split_files(
    output_root: Path,
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> None:
    split_dir = output_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"

    train_lines = [s.image_path.name for s in train_samples]
    val_lines = [s.image_path.name for s in val_samples]

    train_txt.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")


def process_dataset(
    input_root: Path,
    output_root: Path,
    mode: str,
    num_aug: int,
    train_ratio: float,
    overwrite: bool,
    seed: Optional[int],
    limit: Optional[int],
    save_ext: str,
) -> None:
    if mode not in ["dark", "contrast", "clahe", "shadow", "all"]:
        raise ValueError("--mode must be one of: dark, contrast, clahe, shadow, all")

    if not save_ext.startswith("."):
        save_ext = "." + save_ext

    if overwrite and output_root.exists():
        print(f"[INFO] remove existing output: {output_root}")
        shutil.rmtree(output_root)

    dirs = ensure_output_dirs(output_root)

    samples = collect_samples(input_root)

    if limit is not None and limit > 0:
        samples = samples[:limit]

    train_samples, val_samples = stratified_split(
        samples=samples,
        train_ratio=train_ratio,
        seed=seed,
    )

    print("=" * 70)
    print("Light Augmentation Split v1")
    print("=" * 70)
    print(f"input_root : {input_root}")
    print(f"output_root: {output_root}")
    print(f"mode       : {mode}")
    print(f"num_aug    : {num_aug}")
    print(f"train_ratio: {train_ratio}")
    print(f"seed       : {seed}")
    print(f"save_ext   : {save_ext}")
    print(f"train base : {len(train_samples)}")
    print(f"val base   : {len(val_samples)}")
    print("=" * 70)

    # 1) val 원본 복사
    for idx, sample in enumerate(val_samples, start=1):
        copy_original_sample(
            sample,
            out_img_dir=dirs["img_val"],
            out_lbl_dir=dirs["lbl_val"],
        )

        if idx % 100 == 0 or idx == len(val_samples):
            print(f"[VAL copy] {idx:5d}/{len(val_samples):5d}")

    # 2) train 원본 복사 + train 증강 생성
    total_train_original = 0
    total_train_aug = 0

    for idx, sample in enumerate(train_samples, start=1):
        copy_original_sample(
            sample,
            out_img_dir=dirs["img_train"],
            out_lbl_dir=dirs["lbl_train"],
        )
        total_train_original += 1

        img = imread_bgr(sample.image_path)

        for aug_idx in range(1, num_aug + 1):
            aug_img = apply_augmentation(mode, img)
            save_augmented_sample(
                sample=sample,
                aug_img=aug_img,
                out_img_dir=dirs["img_train"],
                out_lbl_dir=dirs["lbl_train"],
                mode=mode,
                aug_idx=aug_idx,
                save_ext=save_ext,
            )
            total_train_aug += 1

        if idx % 50 == 0 or idx == len(train_samples):
            print(
                f"[TRAIN] {idx:5d}/{len(train_samples):5d} | "
                f"original={total_train_original}, aug={total_train_aug}"
            )

    write_split_files(output_root, train_samples, val_samples)

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"train originals : {total_train_original}")
    print(f"train augmented : {total_train_aug}")
    print(f"val originals   : {len(val_samples)}")
    print(f"output images   : {output_root / IMAGE_DIR_NAME}")
    print(f"output labels   : {output_root / LABEL_DIR_NAME}")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO-seg train/val split + light augmentation script v1"
    )

    parser.add_argument(
        "--input-root",
        type=str,
        default=str(INPUT_ROOT),
        help="입력 dataset root. 내부에 images/, labels/가 있어야 함.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="출력 dataset root.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["dark", "contrast", "clahe", "shadow", "all"],
        help="밝기 증강 모드.",
    )

    parser.add_argument(
        "--num-aug",
        type=int,
        default=2,
        help="train 원본 이미지 1장당 만들 증강 이미지 개수.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="train split 비율. 기본 0.8",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 output-root가 있으면 삭제 후 새로 생성.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="디버깅용. 앞에서 N개 샘플만 사용.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="random seed. 매번 다르게 하려면 -1.",
    )

    parser.add_argument(
        "--save-ext",
        type=str,
        default=AUG_SAVE_EXT,
        help="증강 이미지 저장 확장자. 기본 .jpg",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = None if args.seed == -1 else args.seed
    set_seed(seed)

    process_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        mode=args.mode,
        num_aug=args.num_aug,
        train_ratio=args.train_ratio,
        overwrite=args.overwrite,
        seed=seed,
        limit=args.limit,
        save_ext=args.save_ext,
    )


if __name__ == "__main__":
    main()
