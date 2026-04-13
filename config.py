"""
로컬 환경에 맞게 여기만 수정하면 됩니다.
다른 PC에서는 프로젝트 루트만 옮기고, 필요 시 경로·가중치 파일명을 바꾸세요.

환경 변수로 덮어쓸 수 있습니다 (선택):
  ANNOTATION_PROJECT_ROOT, ANNOTATION_WORKSPACE, ANNOTATION_WEIGHTS_DIR
  ANNOTATION_WEB_HOST, ANNOTATION_WEB_PORT
"""
from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _str_from_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# 프로젝트 루트 (이 파일이 있는 폴더)
PROJECT_ROOT = _path_from_env(
    "ANNOTATION_PROJECT_ROOT",
    Path(__file__).resolve().parent,
)

# 데이터·중간 산출물 루트
WORKSPACE_ROOT = _path_from_env(
    "ANNOTATION_WORKSPACE",
    PROJECT_ROOT / "create",
)

WEIGHTS_DIR = _path_from_env(
    "ANNOTATION_WEIGHTS_DIR",
    PROJECT_ROOT / "weights",
)

# --- Web GUI 설정 ---
WEB_HOST = _str_from_env("ANNOTATION_WEB_HOST", "0.0.0.0")
WEB_PORT = _int_from_env("ANNOTATION_WEB_PORT", 7860)

# --- ROS bag → 프레임 이미지 ---
BAG_DIR = PROJECT_ROOT / "bag"

# 입력 루트
INPUT_DIR = WORKSPACE_ROOT / "input"

# RGB 이미지 저장 폴더
INPUT_IMAGES_DIR = INPUT_DIR / "images"
# RGB-D 원본 저장 폴더 (.npz)
INPUT_RGBD_DIR = INPUT_DIR / "rgbd"

# 하위 호환용 별칭 (기존 코드 깨짐 방지)
IMAGES_RAW_DIR = INPUT_IMAGES_DIR

# 1차 자동 마스킹 (YOLO-seg + SAM) 결과
STAGE1_DIR = WORKSPACE_ROOT / "output_1"
STAGE1_MASKS = STAGE1_DIR / "masks"
STAGE1_LABELS = STAGE1_DIR / "labels"
STAGE1_OVERLAYS = STAGE1_DIR / "images"

# Web GUI 정제 결과 (최종 라벨)
STAGE2_DIR = WORKSPACE_ROOT / "output_2"
STAGE2_IMAGES = STAGE2_DIR / "images"
STAGE2_MASKS = STAGE2_DIR / "masks"
STAGE2_LABELS = STAGE2_DIR / "labels"

# export_dataset 이 모으는 폴더
DATASET_DIR = WORKSPACE_ROOT / "dataset"
DATASET_IMAGES = DATASET_DIR / "images"
DATASET_MASKS = DATASET_DIR / "masks"
DATASET_LABELS = DATASET_DIR / "labels"
DATASET_RGBD = DATASET_DIR / "rgbd"

# build_train_split 결과 (Ultralytics YOLO 학습용)
TRAINING_DIR = WORKSPACE_ROOT / "training"

# --- 모델 가중치 ---
YOLO_WEIGHT = WEIGHTS_DIR / "yolo26l-seg_finetuned_mix_2000.pt"
SAM_WEIGHT = WEIGHTS_DIR / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# --- auto_segment (1차 마스킹) 하이퍼파라미터 ---
SEG_CONF_THRESHOLD = 0.01
SEG_BBOX_EXPAND = 1.15
SEG_BATCH_SIZE = 16

# --- build_train_split ---
DEFAULT_VAL_RATIO = 0.1

# --- stack_trim_dataset: 배경별 클래스당 유지 개수 ---
STACK_TRIM_KEEP_PER_BG: dict[str, int] = {
    "paper": 17,
    "floor": 17,
    "desk": 16,
}


def ensure_stage_dirs() -> None:
    """필요한 출력 폴더 생성."""
    for p in (
        INPUT_IMAGES_DIR,
        INPUT_RGBD_DIR,
        STAGE1_MASKS,
        STAGE1_LABELS,
        STAGE1_OVERLAYS,
        STAGE2_IMAGES,
        STAGE2_MASKS,
        STAGE2_LABELS,
        DATASET_IMAGES,
        DATASET_MASKS,
        DATASET_LABELS,
        DATASET_RGBD,
        TRAINING_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)