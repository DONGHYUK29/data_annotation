# -*- coding: utf-8 -*-
"""
augment_training_dataset.py

Unified augmentation script for YOLO-seg training datasets.

Input structure:
  INPUT_ROOT/
    images/train/
    images/val/
    labels/train/
    labels/val/
    dataset.yaml

Output structure:
  OUTPUT_ROOT/
    images/train/   # original train + augmented train
    images/val/     # original val only
    labels/train/
    labels/val/
    dataset.yaml
    aug_meta.json

Modes:
  background : keep object area, augment background only using YOLO-seg polygon mask
  light      : augment whole image brightness/contrast/gamma/CLAHE/shadow
  position   : crop object by mask, paste to background frames with scale/position changes

Typical usage:
  python create/augment_training_dataset.py \
    --input-root data/training \
    --output-root data/training_bg_aug \
    --mode background --submode all --num-aug 2 --overwrite

  python create/augment_training_dataset.py \
    --input-root data/training_bg_aug \
    --output-root data/training_bg_light_aug \
    --mode light --submode all --num-aug 1 --overwrite

  python create/augment_training_dataset.py \
    --input-root data/training_bg_light_aug \
    --output-root data/training_bg_light_pos_aug \
    --mode position --num-aug 1 \
    --bg-root data/backgrounds_lab \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# Constants
# ============================================================

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

BACKGROUND_SUBMODES = {"all", "bg_color", "hard_bg", "clutter"}
LIGHT_SUBMODES = {"all", "dark", "contrast", "clahe", "shadow", "highlight"}

BG_ALIASES = {
    "desk": "desk",
    "table": "desk",
    "책상": "desk",
    "paper": "paper",
    "dohwaji": "paper",
    "도화지": "paper",
    "floor": "floor",
    "ground": "floor",
    "바닥": "floor",
}


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    stem: str
    suffix: str


# ============================================================
# Generic utils
# ============================================================

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def normalize_bg_name(name: str) -> str:
    return BG_ALIASES.get(name.lower(), name.lower())


def infer_bg_from_stem(stem: str) -> str:
    """
    Current dataset names usually begin with:
      desk_5_101.png
      paper_3_1_aug_light_all_01.jpg
      floor_6_120_aug_bg_all_01.jpg

    For augmented names, the first token still identifies the background.
    """
    if not stem:
        return "unknown"
    return normalize_bg_name(stem.split("_")[0])


def find_images(image_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(image_dir.glob(f"*{ext}"))
        paths.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def imwrite_bgr(path: Path, img: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
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


def validate_training_root(input_root: Path) -> None:
    required = [
        input_root / "images" / "train",
        input_root / "images" / "val",
        input_root / "labels" / "train",
        input_root / "labels" / "val",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"required path not found: {p}")


def collect_split_samples(input_root: Path, split: str) -> List[Sample]:
    image_dir = input_root / "images" / split
    label_dir = input_root / "labels" / split

    samples: List[Sample] = []
    for img_path in find_images(image_dir):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] label missing, skip: {img_path.name}")
            continue
        samples.append(Sample(img_path, label_path, img_path.stem, img_path.suffix.lower()))

    return samples


def prepare_output(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if input_root.resolve() == output_root.resolve():
        raise RuntimeError("input-root and output-root must be different.")

    if overwrite and output_root.exists():
        print(f"[INFO] remove existing output: {output_root}")
        shutil.rmtree(output_root)

    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_original_splits(
    train_samples: List[Sample],
    val_samples: List[Sample],
    output_root: Path,
) -> Dict[str, int]:
    counts = {
        "train_images": 0,
        "train_labels": 0,
        "val_images": 0,
        "val_labels": 0,
    }

    for s in train_samples:
        copy_file(s.image_path, output_root / "images" / "train" / s.image_path.name)
        copy_file(s.label_path, output_root / "labels" / "train" / s.label_path.name)
        counts["train_images"] += 1
        counts["train_labels"] += 1

    for s in val_samples:
        copy_file(s.image_path, output_root / "images" / "val" / s.image_path.name)
        copy_file(s.label_path, output_root / "labels" / "val" / s.label_path.name)
        counts["val_images"] += 1
        counts["val_labels"] += 1

    return counts


def write_dataset_yaml(input_root: Path, output_root: Path, num_classes: Optional[int]) -> None:
    input_yaml = input_root / "dataset.yaml"
    output_yaml = output_root / "dataset.yaml"

    if input_yaml.exists():
        lines = input_yaml.read_text(encoding="utf-8").splitlines()
        new_lines: List[str] = []
        has_path = False
        for line in lines:
            if line.strip().startswith("path:"):
                new_lines.append(f"path: {output_root.as_posix()}")
                has_path = True
            else:
                new_lines.append(line)
        if not has_path:
            new_lines.insert(0, f"path: {output_root.as_posix()}")
        output_yaml.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return

    if num_classes is None:
        num_classes = 10

    names = [f"class_{i}" for i in range(num_classes)]
    with output_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {output_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {num_classes}\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")


def save_meta(output_root: Path, meta: Dict) -> None:
    (output_root / "aug_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# YOLO-seg label mask utils
# ============================================================

def parse_yolo_seg_label(label_path: Path, w: int, h: int) -> List[Tuple[int, np.ndarray]]:
    """
    YOLO segmentation format:
      class x1 y1 x2 y2 ... xn yn
    Coordinates are normalized to [0, 1].
    """
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

        if pts.shape[0] >= 3:
            objects.append((cls_id, pts.astype(np.float32)))

    return objects


def build_object_mask_from_objects(objects: List[Tuple[int, np.ndarray]], w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, pts in objects:
        poly = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def build_object_mask(label_path: Path, w: int, h: int) -> np.ndarray:
    objects = parse_yolo_seg_label(label_path, w, h)
    return build_object_mask_from_objects(objects, w, h)


def make_alpha_from_mask(mask: np.ndarray, feather_kernel: int = 3) -> np.ndarray:
    alpha = mask.astype(np.float32) / 255.0
    if feather_kernel and feather_kernel > 1:
        k = feather_kernel if feather_kernel % 2 == 1 else feather_kernel + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha[..., None]


def composite_keep_object(original: np.ndarray, aug_bg_img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    alpha = make_alpha_from_mask(object_mask, feather_kernel=3)
    out = original.astype(np.float32) * alpha + aug_bg_img.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def protect_object_edge(mask: np.ndarray, dilate_iter: int = 2) -> np.ndarray:
    if dilate_iter <= 0:
        return mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=dilate_iter)


def safe_background_mask(object_mask: np.ndarray) -> np.ndarray:
    protected = protect_object_edge(object_mask, dilate_iter=2)
    return (protected == 0).astype(np.uint8) * 255


# ============================================================
# Background mode
# ============================================================

def adjust_hsv_background(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue_shift = random.uniform(-8, 8)
    sat_scale = random.uniform(0.75, 1.25)
    val_scale = random.uniform(0.75, 1.25)
    val_bias = random.uniform(-18, 18)

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale + val_bias, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = random.uniform(0.85, 1.18)
    beta = random.uniform(-10, 10)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.65:
        tint = np.array([
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
        ], dtype=np.float32)
        out = np.clip(out.astype(np.float32) + tint, 0, 255).astype(np.uint8)

    return out


def augment_bg_color(img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    return composite_keep_object(img, adjust_hsv_background(img), object_mask)


def get_mean_color(img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    idx = mask > 0
    if idx.sum() < 10:
        return None
    return img[idx].astype(np.float32).mean(axis=0)


def create_low_contrast_background(img: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    obj_mean = get_mean_color(img, object_mask)
    if obj_mean is None:
        return adjust_hsv_background(img)

    offset = np.array([
        random.uniform(-25, 25),
        random.uniform(-25, 25),
        random.uniform(-25, 25),
    ], dtype=np.float32)
    target_color = np.clip(obj_mean + offset, 0, 255)

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=random.uniform(6, 14))
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_norm = (gray - gray.mean()) / (gray.std() + 1e-6)
    texture_strength = random.uniform(3, 12)

    target = np.ones_like(img, dtype=np.float32) * target_color.reshape(1, 1, 3)
    texture = gray_norm[..., None] * texture_strength
    target = np.clip(target + texture, 0, 255).astype(np.uint8)

    mix = random.uniform(0.55, 0.85)
    aug_bg = np.clip(
        img.astype(np.float32) * (1.0 - mix) + target.astype(np.float32) * mix,
        0,
        255,
    ).astype(np.uint8)

    return composite_keep_object(img, aug_bg, object_mask)


def random_background_rect(bg_mask: np.ndarray, min_frac: float = 0.08, max_frac: float = 0.28) -> Optional[Tuple[int, int, int, int]]:
    h, w = bg_mask.shape[:2]

    for _ in range(80):
        if random.random() < 0.75:
            y1_min, y1_max = 0, max(1, int(h * 0.45))
        else:
            y1_min, y1_max = 0, max(1, int(h * 0.75))

        rw = int(w * random.uniform(min_frac, max_frac))
        rh = int(h * random.uniform(min_frac, max_frac))
        if rw < 8 or rh < 8:
            continue

        x1 = random.randint(0, max(0, w - rw))
        y1 = random.randint(y1_min, max(y1_min, min(y1_max, h - rh)))
        x2 = x1 + rw
        y2 = y1 + rh

        patch_mask = bg_mask[y1:y2, x1:x2]
        bg_ratio = float((patch_mask > 0).sum()) / float(patch_mask.size + 1e-6)

        if bg_ratio > 0.95:
            return x1, y1, x2, y2

    return None


def add_soft_rect_texture(img: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    rect = random_background_rect(bg_mask)
    if rect is None:
        return out

    x1, y1, x2, y2 = rect
    ph, pw = y2 - y1, x2 - x1
    patch = out[y1:y2, x1:x2].copy().astype(np.float32)

    mode = random.choice(["solid", "noise", "line", "blur_patch", "shadow"])

    if mode == "solid":
        color = np.array([
            random.uniform(40, 220),
            random.uniform(40, 220),
            random.uniform(40, 220),
        ], dtype=np.float32)
        synthetic = np.ones_like(patch) * color.reshape(1, 1, 3)

    elif mode == "noise":
        base_color = patch.reshape(-1, 3).mean(axis=0)
        noise = np.random.normal(0, random.uniform(8, 22), size=patch.shape).astype(np.float32)
        synthetic = np.ones_like(patch) * base_color.reshape(1, 1, 3) + noise

    elif mode == "line":
        synthetic = patch.copy()
        line_color = np.array([
            random.uniform(30, 230),
            random.uniform(30, 230),
            random.uniform(30, 230),
        ], dtype=np.float32)
        for _ in range(random.randint(2, 6)):
            p1 = (random.randint(0, pw - 1), random.randint(0, ph - 1))
            p2 = (random.randint(0, pw - 1), random.randint(0, ph - 1))
            cv2.line(synthetic, p1, p2, line_color.tolist(), thickness=random.randint(1, 3))

    elif mode == "blur_patch":
        h, w = img.shape[:2]
        sx1 = random.randint(0, max(0, w - pw))
        sy1 = random.randint(0, max(0, h - ph))
        synthetic = img[sy1:sy1 + ph, sx1:sx1 + pw].copy().astype(np.float32)
        synthetic = cv2.GaussianBlur(synthetic, (0, 0), sigmaX=random.uniform(1.5, 4.5))

    else:
        synthetic = patch * random.uniform(0.45, 0.8)

    synthetic = np.clip(synthetic, 0, 255).astype(np.float32)

    k = max(7, int(min(ph, pw) * 0.18))
    if k % 2 == 0:
        k += 1

    dist = np.zeros((ph, pw), dtype=np.float32)
    cv2.rectangle(dist, (2, 2), (pw - 3, ph - 3), 1.0, thickness=-1)
    dist = cv2.GaussianBlur(dist, (k, k), 0)
    dist = np.clip(dist, 0.0, 1.0)

    strength = random.uniform(0.25, 0.65)
    alpha3 = (dist * strength)[..., None]

    blended = patch * (1.0 - alpha3) + synthetic * alpha3
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def augment_clutter(img: np.ndarray, object_mask: np.ndarray, n_items: Optional[int] = None) -> np.ndarray:
    bg_mask = safe_background_mask(object_mask)
    out = img.copy()

    if n_items is None:
        n_items = random.randint(1, 4)

    for _ in range(n_items):
        out = add_soft_rect_texture(out, bg_mask)

    return composite_keep_object(img, out, object_mask)


def augment_background(img: np.ndarray, object_mask: np.ndarray, submode: str) -> np.ndarray:
    if submode == "bg_color":
        return augment_bg_color(img, object_mask)
    if submode == "hard_bg":
        return create_low_contrast_background(img, object_mask)
    if submode == "clutter":
        return augment_clutter(img, object_mask)
    if submode == "all":
        out = img.copy()
        r = random.random()
        if r < 0.45:
            out = augment_bg_color(out, object_mask)
        elif r < 0.85:
            out = create_low_contrast_background(out, object_mask)

        if random.random() < 0.55:
            out = augment_clutter(out, object_mask, n_items=random.randint(1, 3))

        return composite_keep_object(img, out, object_mask)

    raise ValueError(f"unknown background submode: {submode}")


# ============================================================
# Light mode
# ============================================================

def clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    out = img.astype(np.float32)
    out = (out - 127.5) * contrast + 127.5 + brightness
    return clip_uint8(out)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    return cv2.LUT(img, clip_uint8(table))


def apply_clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8, blend: float = 0.7) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    out = img.astype(np.float32) * (1.0 - blend) + out.astype(np.float32) * blend
    return clip_uint8(out)


def recover_if_too_dark(img: np.ndarray, min_mean: float = 55.0, target_mean: float = 70.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    if mean_val >= min_mean:
        return img

    gain = target_mean / max(mean_val, 1.0)
    gain = min(gain, 1.45)
    return clip_uint8(img.astype(np.float32) * gain)


def apply_random_shadow(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

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
        center = (random.randint(0, max(0, w - 1)), random.randint(0, max(0, h - 1)))
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
    return clip_uint8(out * factor)


def apply_mild_highlight(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    center = (random.randint(0, max(0, w - 1)), random.randint(0, max(0, h - 1)))
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
    return clip_uint8(out + mask[..., None] * strength)


def augment_dark(img: np.ndarray) -> np.ndarray:
    contrast = random.uniform(0.90, 1.12)
    brightness = random.uniform(-42, -8)
    out = adjust_brightness_contrast(img, brightness=brightness, contrast=contrast)

    if random.random() < 0.65:
        gamma = random.uniform(1.08, 1.45)
        out = apply_gamma(out, gamma)

    out = recover_if_too_dark(out, min_mean=48.0, target_mean=63.0)

    if random.random() < 0.35:
        out = apply_clahe_lab(
            out,
            clip_limit=random.uniform(1.3, 2.2),
            tile_grid_size=8,
            blend=random.uniform(0.20, 0.45),
        )
    return out


def augment_contrast(img: np.ndarray) -> np.ndarray:
    brightness = random.uniform(-35, 25)
    contrast = random.uniform(0.65, 1.35)
    out = adjust_brightness_contrast(img, brightness=brightness, contrast=contrast)

    if random.random() < 0.5:
        gamma = random.uniform(0.75, 1.35)
        out = apply_gamma(out, gamma)

    return out


def augment_clahe(img: np.ndarray) -> np.ndarray:
    out = apply_clahe_lab(
        img,
        clip_limit=random.uniform(1.5, 3.0),
        tile_grid_size=random.choice([8, 8, 12]),
        blend=random.uniform(0.45, 0.75),
    )

    if random.random() < 0.5:
        out = adjust_brightness_contrast(
            out,
            brightness=random.uniform(-25, 10),
            contrast=random.uniform(0.90, 1.15),
        )
    return out


def augment_shadow(img: np.ndarray) -> np.ndarray:
    out = img.copy()

    if random.random() < 0.5:
        out = adjust_brightness_contrast(
            out,
            brightness=random.uniform(-18, -3),
            contrast=random.uniform(0.92, 1.08),
        )

    out = apply_random_shadow(out)
    return recover_if_too_dark(out, min_mean=55.0, target_mean=68.0)


def augment_light(img: np.ndarray, submode: str) -> np.ndarray:
    if submode == "dark":
        return augment_dark(img)
    if submode == "contrast":
        return augment_contrast(img)
    if submode == "clahe":
        return augment_clahe(img)
    if submode == "shadow":
        return augment_shadow(img)
    if submode == "highlight":
        return apply_mild_highlight(img)
    if submode == "all":
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
            out = apply_mild_highlight(img)
            mode_name = "highlight"

        if mode_name != "clahe" and random.random() < 0.15:
            out = apply_clahe_lab(
                out,
                clip_limit=random.uniform(1.3, 2.2),
                tile_grid_size=8,
                blend=random.uniform(0.25, 0.45),
            )
        return out

    raise ValueError(f"unknown light submode: {submode}")


# ============================================================
# Position mode
# ============================================================

def load_backgrounds(bg_root: Path) -> Dict[str, List[Path]]:
    if not bg_root.exists():
        raise FileNotFoundError(f"bg-root not found: {bg_root}")

    bg_dict: Dict[str, List[Path]] = {}
    for d in sorted(bg_root.iterdir()):
        if not d.is_dir():
            continue
        key = normalize_bg_name(d.name)
        paths = find_images(d)
        if paths:
            bg_dict.setdefault(key, []).extend(paths)
            print(f"[BG] {key}: {len(paths)} images")

    if not bg_dict:
        raise RuntimeError(f"No background images found in {bg_root}")

    return bg_dict


def pick_background(bg_dict: Dict[str, List[Path]], src_bg: str, same_bg_only: bool) -> Path:
    key = normalize_bg_name(src_bg)

    if key in bg_dict and bg_dict[key]:
        return random.choice(bg_dict[key])

    if same_bg_only:
        # fallback because some augmented names may have unusual prefixes
        print(f"[WARN] no matching bg for '{src_bg}', fallback to any background")

    all_paths: List[Path] = []
    for paths in bg_dict.values():
        all_paths.extend(paths)
    return random.choice(all_paths)


def bbox_from_mask(mask: np.ndarray, pad: int = 4) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 10 or len(ys) < 10:
        return None

    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + 1 + pad)
    y2 = min(h, int(ys.max()) + 1 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def choose_scale(far_prob: float = 0.60) -> float:
    r = random.random()
    if r < far_prob:
        return random.uniform(0.38, 0.75)
    if r < far_prob + 0.20:
        return random.uniform(0.80, 1.15)
    return random.uniform(1.20, 1.65)


def choose_position(canvas_w: int, canvas_h: int, obj_w: int, obj_h: int, edge_prob: float = 0.70) -> Tuple[int, int]:
    max_x = max(0, canvas_w - obj_w)
    max_y = max(0, canvas_h - obj_h)

    if max_x == 0:
        x = 0
    elif random.random() < edge_prob:
        edge_width = int(canvas_w * 0.22)
        if random.choice(["left", "right"]) == "left":
            x = random.randint(0, min(max_x, max(1, edge_width)))
        else:
            x = random.randint(max(0, max_x - edge_width), max_x)
    else:
        x = random.randint(0, max_x)

    if max_y == 0:
        y = 0
    elif random.random() < edge_prob:
        edge_height = int(canvas_h * 0.22)
        if random.choice(["top", "bottom"]) == "top":
            y = random.randint(0, min(max_y, max(1, edge_height)))
        else:
            y = random.randint(max(0, max_y - edge_height), max_y)
    else:
        y = random.randint(0, max_y)

    return x, y


def adjust_object_to_bg(obj: np.ndarray, bg_crop: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    obj_f = obj.astype(np.float32)
    a = alpha[..., 0] > 0.2
    if a.sum() < 20:
        return obj

    obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    obj_mean = float(obj_gray[a].mean())
    bg_mean = float(bg_gray.mean())

    target_ratio = bg_mean / max(obj_mean, 1.0)
    target_ratio = np.clip(target_ratio, 0.75, 1.25)

    strength = 0.35
    ratio = 1.0 * (1.0 - strength) + target_ratio * strength

    out = obj_f * ratio
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_labels(
    objects: List[Tuple[int, np.ndarray]],
    crop_box: Tuple[int, int, int, int],
    scale: float,
    paste_x: int,
    paste_y: int,
    out_w: int,
    out_h: int,
) -> List[str]:
    x1, y1, _, _ = crop_box
    lines: List[str] = []

    for cls, pts in objects:
        new_pts = pts.copy()
        new_pts[:, 0] = (new_pts[:, 0] - x1) * scale + paste_x
        new_pts[:, 1] = (new_pts[:, 1] - y1) * scale + paste_y

        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, out_w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, out_h - 1)

        if new_pts.shape[0] < 3:
            continue

        coords: List[str] = []
        for x, y in new_pts:
            coords.append(f"{x / out_w:.6f}")
            coords.append(f"{y / out_h:.6f}")

        lines.append(f"{cls} " + " ".join(coords))

    return lines


def augment_position(
    img: np.ndarray,
    label_path: Path,
    bg_path: Path,
    edge_prob: float,
    far_prob: float,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    h, w = img.shape[:2]
    objects = parse_yolo_seg_label(label_path, w, h)
    if not objects:
        return None

    mask = build_object_mask_from_objects(objects, w, h)
    crop_box = bbox_from_mask(mask, pad=4)
    if crop_box is None:
        return None

    x1, y1, x2, y2 = crop_box
    obj_crop = img[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].copy()

    bg = imread_bgr(bg_path)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)

    for _ in range(30):
        scale = choose_scale(far_prob=far_prob)
        new_w = max(4, int(round(obj_crop.shape[1] * scale)))
        new_h = max(4, int(round(obj_crop.shape[0] * scale)))
        if new_w < w * 0.92 and new_h < h * 0.92:
            break
    else:
        scale = min((w * 0.85) / obj_crop.shape[1], (h * 0.85) / obj_crop.shape[0])
        new_w = max(4, int(round(obj_crop.shape[1] * scale)))
        new_h = max(4, int(round(obj_crop.shape[0] * scale)))

    obj_rs = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    alpha = make_alpha_from_mask(mask_rs, feather_kernel=3)

    px, py = choose_position(w, h, new_w, new_h, edge_prob=edge_prob)

    bg_roi = bg[py:py + new_h, px:px + new_w].copy()
    obj_rs = adjust_object_to_bg(obj_rs, bg_roi, alpha)

    comp_roi = obj_rs.astype(np.float32) * alpha + bg_roi.astype(np.float32) * (1.0 - alpha)
    out = bg.copy()
    out[py:py + new_h, px:px + new_w] = np.clip(comp_roi, 0, 255).astype(np.uint8)

    label_lines = transform_labels(
        objects=objects,
        crop_box=crop_box,
        scale=scale,
        paste_x=px,
        paste_y=py,
        out_w=w,
        out_h=h,
    )

    if not label_lines:
        return None

    return out, label_lines


# ============================================================
# Main processing
# ============================================================

def make_aug_stem(stem: str, mode: str, submode: str, aug_idx: int) -> str:
    if mode == "background":
        return f"{stem}_aug_bg_{submode}_{aug_idx:02d}"
    if mode == "light":
        return f"{stem}_aug_light_{submode}_{aug_idx:02d}"
    if mode == "position":
        return f"{stem}_aug_pos_{aug_idx:02d}"
    raise ValueError(mode)


def process(
    input_root: Path,
    output_root: Path,
    mode: str,
    submode: str,
    num_aug: int,
    overwrite: bool,
    seed: Optional[int],
    save_ext: str,
    jpeg_quality: int,
    bg_root: Optional[Path],
    same_bg_only: bool,
    edge_prob: float,
    far_prob: float,
    limit: Optional[int],
    num_classes: Optional[int],
) -> None:
    validate_training_root(input_root)

    if not save_ext.startswith("."):
        save_ext = "." + save_ext

    if mode == "background" and submode not in BACKGROUND_SUBMODES:
        raise ValueError(f"background submode must be one of {sorted(BACKGROUND_SUBMODES)}")
    if mode == "light" and submode not in LIGHT_SUBMODES:
        raise ValueError(f"light submode must be one of {sorted(LIGHT_SUBMODES)}")
    if mode == "position" and bg_root is None:
        raise ValueError("position mode requires --bg-root")

    set_seed(seed)
    prepare_output(input_root, output_root, overwrite=overwrite)

    train_samples = collect_split_samples(input_root, "train")
    val_samples = collect_split_samples(input_root, "val")

    if limit is not None and limit > 0:
        train_samples = train_samples[:limit]

    copy_counts = copy_original_splits(train_samples, val_samples, output_root)

    bg_dict: Optional[Dict[str, List[Path]]] = None
    if mode == "position":
        assert bg_root is not None
        bg_dict = load_backgrounds(bg_root)

    out_img_train = output_root / "images" / "train"
    out_lbl_train = output_root / "labels" / "train"

    total_aug = 0
    failed = 0

    print("=" * 80)
    print("Unified YOLO-seg Training Augmentation")
    print("=" * 80)
    print(f"input_root  : {input_root}")
    print(f"output_root : {output_root}")
    print(f"mode        : {mode}")
    print(f"submode     : {submode}")
    print(f"num_aug     : {num_aug}")
    print(f"train base  : {len(train_samples)}")
    print(f"val copied  : {len(val_samples)}")
    print(f"save_ext    : {save_ext}")
    print(f"seed        : {seed}")
    if bg_root is not None:
        print(f"bg_root     : {bg_root}")
    print("=" * 80)

    for idx, sample in enumerate(train_samples, start=1):
        try:
            img = imread_bgr(sample.image_path)
        except Exception as e:
            print(f"[WARN] image read failed, skip augment: {sample.image_path} | {e}")
            failed += num_aug
            continue

        h, w = img.shape[:2]

        for aug_idx in range(1, num_aug + 1):
            aug_stem = make_aug_stem(sample.stem, mode, submode, aug_idx)
            out_img_path = out_img_train / f"{aug_stem}{save_ext}"
            out_lbl_path = out_lbl_train / f"{aug_stem}.txt"

            try:
                if mode == "background":
                    object_mask = build_object_mask(sample.label_path, w, h)
                    if (object_mask > 0).sum() < 10:
                        print(f"[WARN] empty mask, skip augmentation: {sample.label_path.name}")
                        failed += 1
                        continue
                    aug_img = augment_background(img, object_mask, submode=submode)
                    imwrite_bgr(out_img_path, aug_img, jpeg_quality=jpeg_quality)
                    copy_file(sample.label_path, out_lbl_path)

                elif mode == "light":
                    aug_img = augment_light(img, submode=submode)
                    imwrite_bgr(out_img_path, aug_img, jpeg_quality=jpeg_quality)
                    copy_file(sample.label_path, out_lbl_path)

                elif mode == "position":
                    assert bg_dict is not None
                    src_bg = infer_bg_from_stem(sample.stem)
                    bg_path = pick_background(bg_dict, src_bg=src_bg, same_bg_only=same_bg_only)
                    result = augment_position(
                        img=img,
                        label_path=sample.label_path,
                        bg_path=bg_path,
                        edge_prob=edge_prob,
                        far_prob=far_prob,
                    )
                    if result is None:
                        print(f"[WARN] position aug failed: {sample.image_path.name}")
                        failed += 1
                        continue

                    aug_img, label_lines = result
                    imwrite_bgr(out_img_path, aug_img, jpeg_quality=jpeg_quality)
                    out_lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

                else:
                    raise ValueError(mode)

                total_aug += 1

            except Exception as e:
                print(f"[WARN] augmentation failed: {sample.image_path.name} | {e}")
                failed += 1

        if idx % 50 == 0 or idx == len(train_samples):
            print(f"[{idx:5d}/{len(train_samples):5d}] augmented={total_aug}, failed={failed}")

    write_dataset_yaml(input_root, output_root, num_classes=num_classes)

    meta = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mode": mode,
        "submode": submode,
        "num_aug": num_aug,
        "seed": seed,
        "save_ext": save_ext,
        "jpeg_quality": jpeg_quality,
        "train_base": len(train_samples),
        "val_copied": len(val_samples),
        "copied": copy_counts,
        "augmented": total_aug,
        "failed": failed,
        "bg_root": str(bg_root) if bg_root is not None else None,
        "same_bg_only": same_bg_only,
        "edge_prob": edge_prob,
        "far_prob": far_prob,
    }
    save_meta(output_root, meta)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"output dataset.yaml : {output_root / 'dataset.yaml'}")
    print(f"train originals     : {len(train_samples)}")
    print(f"train augmented     : {total_aug}")
    print(f"val originals       : {len(val_samples)}")
    print(f"failed              : {failed}")
    print(f"meta                : {output_root / 'aug_meta.json'}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified YOLO-seg augmentation for training datasets"
    )

    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Input training dataset root with images/train, images/val, labels/train, labels/val.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output dataset root.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["background", "light", "position"],
        help="Augmentation mode.",
    )
    parser.add_argument(
        "--submode",
        type=str,
        default="all",
        help="background: all/bg_color/hard_bg/clutter, light: all/dark/contrast/clahe/shadow/highlight, position: ignored.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=1,
        help="Number of augmented images per train image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-root if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Use -1 for random each run.",
    )
    parser.add_argument(
        "--save-ext",
        type=str,
        default=".jpg",
        help="Augmented image extension. Original images are copied as-is.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for augmented images.",
    )
    parser.add_argument(
        "--bg-root",
        type=str,
        default=None,
        help="Background image root for position mode. Expected subdirs: desk/paper/floor.",
    )
    parser.add_argument(
        "--allow-any-bg",
        action="store_true",
        help="Position mode: allow fallback to any background if matching background is unavailable.",
    )
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.70,
        help="Position mode: probability of placing object near image edge.",
    )
    parser.add_argument(
        "--far-prob",
        type=float,
        default=0.60,
        help="Position mode: probability of using scale-down far-object augmentation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug only. Use first N train samples.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Used only if input-root/dataset.yaml does not exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = None if args.seed == -1 else args.seed

    process(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        mode=args.mode,
        submode=args.submode,
        num_aug=args.num_aug,
        overwrite=args.overwrite,
        seed=seed,
        save_ext=args.save_ext,
        jpeg_quality=args.jpeg_quality,
        bg_root=Path(args.bg_root) if args.bg_root is not None else None,
        same_bg_only=not args.allow_any_bg,
        edge_prob=args.edge_prob,
        far_prob=args.far_prob,
        limit=args.limit,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
