# Data Annotation and Training

## 권장 작업 디렉터리

대용량 데이터는 Git 추적 대상이 아닌 저장소 루트의 `data/` 아래에 둡니다.

```text
data/
├── annotation_work/
│   ├── input/
│   ├── output_1/{images,masks,labels}/
│   └── output_2/{images,masks,labels}/
├── dataset/{images,labels,masks}/
├── backgrounds_lab/{desk,paper,floor}/
└── bags/
```

`data/`와 `weights/`는 `.gitignore`로 제외됩니다.

## 데이터 제작 순서

1. `tools/capture/`로 RGB, RGB-D 또는 RealSense bag에서 프레임을 수집합니다.
2. `pipeline/` 또는 `tools/annotation/generate_dataset.py`로 YOLO + SAM 초기 라벨을 만듭니다.
3. `tools/annotation/GUI.py`나 `pipeline/gui_editor/`에서 마스크를 수정합니다.
4. `export_dataset.py`, `build_dataset.py`로 YOLO-seg 데이터셋을 구성합니다.
5. `tools/augmentation/`으로 배경·조명·위치 증강을 적용합니다.
6. `check_labels.py`와 `tools/evaluation/run_yolo_seg_badcase_simple.py`로 검수합니다.

## 가중치 경로

GUI와 자동 라벨 생성기는 다음 환경변수를 지원합니다.

```bash
export KERI_ANNOTATION_WORKDIR="$PWD/data/annotation_work"
export YOLO_DETECTION_MODEL="/path/to/yolo-detection.pt"
export SAM_CHECKPOINT="/path/to/sam_vit_b_01ec64.pth"
```

자동 라벨 생성:

```bash
python data_annotation/tools/annotation/generate_dataset.py
```

라벨 편집 GUI:

```bash
python data_annotation/tools/annotation/GUI.py
```

## 초기 파이프라인

저장소 루트에서 실행합니다.

```bash
PYTHONPATH=data_annotation/pipeline \
python data_annotation/pipeline/runs/detect_seg.py \
  --input-dir sample_data/rgbd/images \
  --yolo-model /path/to/detection.pt \
  --sam-model /path/to/sam_vit_b_01ec64.pth
```

생성물은 `data_annotation/pipeline/output/`과 `cache/`에 저장되며 Git에서 제외됩니다.

## 대표 증강 명령

```bash
python data_annotation/tools/augmentation/augment_training_dataset.py \
  --input-root data/training \
  --output-root data/training_bg_light_aug \
  --mode light --submode all --num-aug 1 --overwrite
```

각 도구의 전체 인자는 `python <script> --help`로 확인할 수 있습니다.
