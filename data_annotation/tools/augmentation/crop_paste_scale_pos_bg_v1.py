# -*- coding: utf-8 -*-
"""
crop_paste_scale_pos_bg_v1.py

기능:
- 기존 dataset_light_aug_v2를 base로 복사
- 원본 train split에서 object를 YOLO-seg mask 기준으로 crop
- RealSense로 촬영한 background frame에 확대/축소 + 외곽 이동하여 paste
- label polygon도 동일하게 transform
- val은 건드리지 않음

입력:
1) 원본 dataset:
   data/dataset

2) base dataset:
   data/dataset_light_aug_v2

3) background root:
   data/backgrounds_lab
   ├─ paper/
   ├─ desk/
   └─ floor/

출력:
   data/dataset_light_scale_pos_bg_v1
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


SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ObjSample:
    image_path: Path
    label_path: Path
    stem: str
    bg_name: str


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def find_images(d: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(d.glob(f"*{ext}"))
        paths.extend(d.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def imwrite(path: Path, img: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in [".jpg", ".jpeg"]:
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif ext == ".png":
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(ext, img)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    buf.tofile(str(path))


def parse_bg_from_stem(stem: str) -> str:
    """
    예:
      desk_1_3 -> desk
      paper_10_66 -> paper
      floor_3_12 -> floor

    background 이름에 underscore가 있어도 뒤 2개를 class/index로 보고 앞부분을 bg로 사용.
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return "unknown"
    return "_".join(parts[:-2])


def normalize_bg_name(bg: str) -> str:
    """
    필요 시 배경 이름 alias 처리.
    네 데이터가 paper/desk/floor면 그대로 동작.
    """
    b = bg.lower()

    aliases = {
        "paper": ["paper", "paperboard", "white", "dohwaji", "도화지"],
        "desk": ["desk", "table", "책상"],
        "floor": ["floor", "ground", "바닥"],
    }

    for key, vals in aliases.items():
        if b in vals:
            return key

    return b


def collect_original_samples(original_root: Path) -> Dict[str, ObjSample]:
    img_dir = original_root / "images"
    lbl_dir = original_root / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not lbl_dir.exists():
        raise FileNotFoundError(lbl_dir)

    out: Dict[str, ObjSample] = {}

    for img_path in find_images(img_dir):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"[WARN] missing label, skip: {img_path.name}")
            continue

        bg = normalize_bg_name(parse_bg_from_stem(img_path.stem))

        out[img_path.name] = ObjSample(
            image_path=img_path,
            label_path=lbl_path,
            stem=img_path.stem,
            bg_name=bg,
        )

    return out


def load_train_original_names(base_root: Path) -> List[str]:
    """
    light_aug_split_v1.py가 만든 split/train.txt를 우선 사용.
    없으면 images/train에서 _aug_ 없는 원본 파일만 추정.
    """
    split_train = base_root / "split" / "train.txt"

    if split_train.exists():
        names = [
            line.strip()
            for line in split_train.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"[INFO] loaded train split: {split_train} ({len(names)} names)")
        return names

    print("[WARN] split/train.txt not found. Inferring originals from images/train")
    img_train = base_root / "images" / "train"

    names = []
    for p in find_images(img_train):
        if "_aug_" not in p.stem:
            names.append(p.name)

    print(f"[INFO] inferred original train names: {len(names)}")
    return sorted(names)


def parse_yolo_seg(label_path: Path, w: int, h: int) -> List[Tuple[int, np.ndarray]]:
    """
    return: [(cls, pts_pixel_float[N,2]), ...]
    """
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    objs: List[Tuple[int, np.ndarray]] = []

    for line_idx, line in enumerate(text.splitlines()):
        parts = line.strip().split()
        if len(parts) < 7:
            print(f"[WARN] invalid label skip: {label_path.name}:{line_idx+1}")
            continue

        try:
            cls = int(float(parts[0]))
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            print(f"[WARN] parse failed: {label_path.name}:{line_idx+1}")
            continue

        if len(coords) % 2 != 0:
            coords = coords[:-1]

        pts = coords.reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * w
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * h

        if pts.shape[0] >= 3:
            objs.append((cls, pts))

    return objs


def build_union_mask(objs: List[Tuple[int, np.ndarray]], w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    for _, pts in objs:
        poly = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)

    return mask


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


def make_alpha(mask_crop: np.ndarray, feather: int = 3) -> np.ndarray:
    alpha = mask_crop.astype(np.float32) / 255.0

    if feather > 1:
        k = feather if feather % 2 == 1 else feather + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    return alpha[..., None]


def choose_scale() -> float:
    """
    현재 문제는 먼 거리 성능 저하가 크므로 scale-down 비중을 높게 설정.
    """
    r = random.random()

    if r < 0.60:
        # far
        return random.uniform(0.38, 0.75)
    elif r < 0.80:
        # near original
        return random.uniform(0.80, 1.15)
    else:
        # near
        return random.uniform(1.20, 1.65)


def choose_position(canvas_w: int, canvas_h: int, obj_w: int, obj_h: int) -> Tuple[int, int]:
    """
    외곽/모서리 비율을 높게.
    object는 일단 화면 안에 완전히 들어오게 둠.
    """
    max_x = max(0, canvas_w - obj_w)
    max_y = max(0, canvas_h - obj_h)

    if max_x == 0:
        x = 0
    else:
        if random.random() < 0.70:
            # edge placement
            side = random.choice(["left", "right"])
            margin = int(canvas_w * 0.08)
            edge_width = int(canvas_w * 0.22)

            if side == "left":
                x = random.randint(0, min(max_x, max(1, edge_width)))
            else:
                low = max(0, max_x - edge_width)
                x = random.randint(low, max_x)
        else:
            x = random.randint(0, max_x)

    if max_y == 0:
        y = 0
    else:
        if random.random() < 0.70:
            side = random.choice(["top", "bottom"])
            edge_height = int(canvas_h * 0.22)

            if side == "top":
                y = random.randint(0, min(max_y, max(1, edge_height)))
            else:
                low = max(0, max_y - edge_height)
                y = random.randint(low, max_y)
        else:
            y = random.randint(0, max_y)

    return x, y


def adjust_object_to_bg(obj: np.ndarray, bg_crop: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    합성 티를 줄이기 위한 약한 밝기 매칭.
    너무 강하게 하면 object 색이 바뀌므로 보수적으로 적용.
    """
    obj_f = obj.astype(np.float32)
    bg_f = bg_crop.astype(np.float32)

    a = alpha[..., 0] > 0.2
    if a.sum() < 20:
        return obj

    obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    obj_mean = float(obj_gray[a].mean())
    bg_mean = float(bg_gray.mean())

    # 배경과 object 밝기 차이를 일부만 보정
    target_ratio = bg_mean / max(obj_mean, 1.0)
    target_ratio = np.clip(target_ratio, 0.75, 1.25)

    strength = 0.35
    ratio = 1.0 * (1.0 - strength) + target_ratio * strength

    out = obj_f * ratio
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_labels(
    objs: List[Tuple[int, np.ndarray]],
    crop_box: Tuple[int, int, int, int],
    scale: float,
    paste_x: int,
    paste_y: int,
    out_w: int,
    out_h: int,
) -> List[str]:
    x1, y1, _, _ = crop_box

    lines: List[str] = []

    for cls, pts in objs:
        new_pts = pts.copy()
        new_pts[:, 0] = (new_pts[:, 0] - x1) * scale + paste_x
        new_pts[:, 1] = (new_pts[:, 1] - y1) * scale + paste_y

        # 안전 클리핑. v1에서는 object를 화면 안에 넣으므로 큰 문제 없음.
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, out_w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, out_h - 1)

        # 너무 작은 polygon 제외
        if new_pts.shape[0] < 3:
            continue

        nx = new_pts[:, 0] / out_w
        ny = new_pts[:, 1] / out_h

        coords = []
        for x, y in zip(nx, ny):
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")

        lines.append(f"{cls} " + " ".join(coords))

    return lines


def load_backgrounds(bg_root: Path) -> Dict[str, List[Path]]:
    bg_dict: Dict[str, List[Path]] = {}

    for d in sorted(bg_root.iterdir()):
        if not d.is_dir():
            continue

        paths = find_images(d)
        if paths:
            bg_dict[d.name.lower()] = paths
            print(f"[BG] {d.name}: {len(paths)} images")

    if not bg_dict:
        raise RuntimeError(f"No background images found in {bg_root}")

    return bg_dict


def pick_background(bg_dict: Dict[str, List[Path]], bg_name: str) -> Path:
    key = normalize_bg_name(bg_name)

    if key in bg_dict and bg_dict[key]:
        return random.choice(bg_dict[key])

    # fallback: 아무 배경이나 사용
    all_paths = []
    for paths in bg_dict.values():
        all_paths.extend(paths)

    return random.choice(all_paths)


def copy_base_dataset(base_root: Path, output_root: Path, overwrite: bool) -> None:
    if overwrite and output_root.exists():
        print(f"[INFO] remove existing output: {output_root}")
        shutil.rmtree(output_root)

    if output_root.exists():
        print(f"[INFO] output exists, skip base copy: {output_root}")
        return

    print(f"[INFO] copy base dataset")
    print(f"  from: {base_root}")
    print(f"  to  : {output_root}")
    shutil.copytree(base_root, output_root)


def paste_one(
    sample: ObjSample,
    bg_path: Path,
    save_img_path: Path,
    save_lbl_path: Path,
    save_ext_quality: int = 95,
) -> bool:
    src = imread(sample.image_path)
    h, w = src.shape[:2]

    objs = parse_yolo_seg(sample.label_path, w, h)
    if not objs:
        return False

    mask = build_union_mask(objs, w, h)
    crop_box = bbox_from_mask(mask, pad=4)
    if crop_box is None:
        return False

    x1, y1, x2, y2 = crop_box

    obj_crop = src[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].copy()

    bg = imread(bg_path)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)

    # scale retry
    for _ in range(30):
        scale = choose_scale()
        new_w = max(4, int(round(obj_crop.shape[1] * scale)))
        new_h = max(4, int(round(obj_crop.shape[0] * scale)))

        # 너무 큰 경우 재시도
        if new_w < w * 0.92 and new_h < h * 0.92:
            break
    else:
        scale = min((w * 0.85) / obj_crop.shape[1], (h * 0.85) / obj_crop.shape[0])
        new_w = max(4, int(round(obj_crop.shape[1] * scale)))
        new_h = max(4, int(round(obj_crop.shape[0] * scale)))

    obj_rs = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    alpha = make_alpha(mask_rs, feather=3)

    px, py = choose_position(w, h, new_w, new_h)

    bg_roi = bg[py:py + new_h, px:px + new_w].copy()
    obj_rs = adjust_object_to_bg(obj_rs, bg_roi, alpha)

    comp_roi = obj_rs.astype(np.float32) * alpha + bg_roi.astype(np.float32) * (1.0 - alpha)
    out = bg.copy()
    out[py:py + new_h, px:px + new_w] = np.clip(comp_roi, 0, 255).astype(np.uint8)

    label_lines = transform_labels(
        objs=objs,
        crop_box=crop_box,
        scale=scale,
        paste_x=px,
        paste_y=py,
        out_w=w,
        out_h=h,
    )

    if not label_lines:
        return False

    imwrite(save_img_path, out, quality=save_ext_quality)
    save_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    save_lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return True


def make_dataset(
    original_root: Path,
    base_root: Path,
    bg_root: Path,
    output_root: Path,
    num_aug: int,
    overwrite: bool,
    seed: Optional[int],
    limit: Optional[int],
    save_ext: str,
) -> None:
    set_seed(seed)

    if not save_ext.startswith("."):
        save_ext = "." + save_ext

    copy_base_dataset(base_root, output_root, overwrite=overwrite)

    out_img_train = output_root / "images" / "train"
    out_lbl_train = output_root / "labels" / "train"
    out_img_train.mkdir(parents=True, exist_ok=True)
    out_lbl_train.mkdir(parents=True, exist_ok=True)

    bg_dict = load_backgrounds(bg_root)

    original_samples = collect_original_samples(original_root)
    train_names = load_train_original_names(base_root)

    train_samples: List[ObjSample] = []
    for name in train_names:
        if name in original_samples:
            train_samples.append(original_samples[name])
        else:
            stem = Path(name).stem
            found = None
            for p_name, s in original_samples.items():
                if Path(p_name).stem == stem:
                    found = s
                    break
            if found is not None:
                train_samples.append(found)
            else:
                print(f"[WARN] train original not found in original dataset: {name}")

    if limit is not None and limit > 0:
        train_samples = train_samples[:limit]

    print("=" * 70)
    print("Crop-Paste Scale/Position/Background Aug v1")
    print("=" * 70)
    print(f"original_root: {original_root}")
    print(f"base_root    : {base_root}")
    print(f"bg_root      : {bg_root}")
    print(f"output_root  : {output_root}")
    print(f"num_aug      : {num_aug}")
    print(f"train samples: {len(train_samples)}")
    print(f"save_ext     : {save_ext}")
    print("=" * 70)

    total = 0
    failed = 0

    for i, sample in enumerate(train_samples, start=1):
        for aug_idx in range(1, num_aug + 1):
            bg_path = pick_background(bg_dict, sample.bg_name)

            out_stem = f"{sample.stem}_aug_scale_pos_bg_{aug_idx:02d}"
            out_img = out_img_train / f"{out_stem}{save_ext}"
            out_lbl = out_lbl_train / f"{out_stem}.txt"

            ok = paste_one(
                sample=sample,
                bg_path=bg_path,
                save_img_path=out_img,
                save_lbl_path=out_lbl,
            )

            if ok:
                total += 1
            else:
                failed += 1

        if i % 50 == 0 or i == len(train_samples):
            print(f"[{i:5d}/{len(train_samples):5d}] generated={total}, failed={failed}")

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"generated: {total}")
    print(f"failed   : {failed}")
    print(f"output   : {output_root}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset"),
    )
    parser.add_argument(
        "--base-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_light_aug_v2"),
    )
    parser.add_argument(
        "--bg-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "backgrounds_lab"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset_light_scale_pos_bg_v1"),
    )
    parser.add_argument("--num-aug", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-ext", type=str, default=".jpg")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = None if args.seed == -1 else args.seed

    make_dataset(
        original_root=Path(args.original_root),
        base_root=Path(args.base_root),
        bg_root=Path(args.bg_root),
        output_root=Path(args.output_root),
        num_aug=args.num_aug,
        overwrite=args.overwrite,
        seed=seed,
        limit=args.limit,
        save_ext=args.save_ext,
    )


if __name__ == "__main__":
    main()
