"""Ultralytics YOLO 학습 CLI 인자 구성 (run.py / Edit 공통).

model = YOLO(str(model_source)) 에는 사용자가 고른 weights/ 내 .pt 또는 .yaml 만 전달한다.
하드코딩된 기본 체크포인트 문자열은 넣지 않는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def augmentation_overrides(level: str) -> dict[str, Any]:
    level = (level or "medium").lower()

    if level == "low":
        return {
            "degrees": 0.0,
            "translate": 0.02,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }

    if level == "high":
        return {
            "degrees": 10.0,
            "translate": 0.2,
            "scale": 0.9,
            "shear": 2.0,
            "perspective": 0.0005,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.1,
        }

    return {}


def resolve_weights_path(weights_arg: str, weights_dir: Path) -> Path:
    """
    weights_dir 안의 .pt 또는 .yaml 만 허용.
    상대 파일명이면 weights_dir 하위로만 해석한다.
    """
    wd = weights_dir.resolve()
    p = Path(str(weights_arg).strip()).expanduser()

    if p.suffix.lower() not in (".pt", ".yaml", ".yml"):
        raise FileNotFoundError(
            f"가중치/모델 정의는 .pt 또는 .yaml 만 허용합니다: {weights_arg!r}"
        )

    if p.is_file():
        rp = p.resolve()
        try:
            rp.relative_to(wd)
        except ValueError:
            raise FileNotFoundError(
                f"파일은 다음 폴더 안에 있어야 합니다: {wd}"
            ) from None
        return rp

    cand = (wd / p.name).resolve()
    if cand.is_file():
        return cand

    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {cand}")


def _optional_bool(v: Any) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v

    s = str(v).lower().strip()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return None


def _parse_cache(v: Any) -> bool | str | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v

    s = str(v).lower().strip()
    if s in ("true", "1", "yes"):
        return True
    if s == "ram":
        return "ram"
    if s == "disk":
        return "disk"
    if s in ("false", "0", "no"):
        return False
    return s


def train_kwargs_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    aug = augmentation_overrides(getattr(args, "augment", "medium") or "medium")

    kw: dict[str, Any] = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "name": args.name,
        "project": str(args.project),
    }

    kw.update(aug)

    opt_map = [
        ("lr0", "lr0"),
        ("patience", "patience"),
        ("optimizer", "optimizer"),
        ("weight_decay", "weight_decay"),
        ("workers", "workers"),
        ("seed", "seed"),
        ("dropout", "dropout"),
        ("save_period", "save_period"),
    ]

    for attr, key in opt_map:
        v = getattr(args, attr, None)
        if v is not None:
            kw[key] = v

    freeze_layers = getattr(args, "freeze", None)
    if getattr(args, "pretrained_backbone_only", False):
        if freeze_layers is None or freeze_layers == 0:
            kw["freeze"] = 22
    elif freeze_layers is not None and freeze_layers > 0:
        kw["freeze"] = freeze_layers

    amp = _optional_bool(getattr(args, "amp", None))
    if amp is not None:
        kw["amp"] = amp

    cos_lr = _optional_bool(getattr(args, "cos_lr", None))
    if cos_lr is not None:
        kw["cos_lr"] = cos_lr

    cache_val = _parse_cache(getattr(args, "cache", None))
    if cache_val is not None:
        kw["cache"] = cache_val

    resume = getattr(args, "resume", None)
    if resume is not None and resume != "":
        sl = str(resume).strip().lower()
        if sl in ("true", "1", "yes"):
            kw["resume"] = True
        elif sl not in ("false", "0", "no"):
            kw["resume"] = str(resume).strip()

    return kw


def add_train_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="weights/ 폴더 내 .pt 또는 .yaml (예: yolo26l-seg.yaml, finetuned.pt)",
    )
    p.add_argument("--name", "--exp-name", dest="name", type=str, required=True)
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="dataset.yaml 경로 (비우면 cfg.TRAINING_DATASET_YAML)",
    )
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.2)
    p.add_argument("--num-classes", dest="num_classes", type=int, required=True)
    p.add_argument("--project", type=Path, default=None)

    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
    )
    p.add_argument("--freeze", type=int, default=None)
    p.add_argument(
        "--augment",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--amp", default=None)
    p.add_argument("--cos-lr", dest="cos_lr", default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--save-period", dest="save_period", type=int, default=None)
    p.add_argument(
        "--pretrained-backbone-only",
        dest="pretrained_backbone_only",
        action="store_true",
        default=False,
    )
