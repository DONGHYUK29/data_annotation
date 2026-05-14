# Data Annotation SW

YOLO segmentation 데이터셋을 만들고 학습까지 이어가기 위한 로컬 annotation 파이프라인입니다.

현재 코드 기준의 기본 흐름은 다음과 같습니다.

```text
입력 이미지 업로드/배치
  -> YOLO-seg 1차 자동 마스크 생성
  -> Web GUI에서 브러시/지우개/SAM point로 마스크 보정
  -> dataset/으로 export
  -> train/val split 및 YOLO 학습
```

## 주요 기능

- 이미지 입력: `create/input/images/`에 PNG/JPG/JPEG/BMP 파일 배치 또는 Web GUI 업로드
- 자동 세그멘테이션: Ultralytics YOLO-seg 모델로 1차 마스크 생성
- 수동 보정 GUI: 브러시, 지우개, undo, SAM point assist 지원
- 결과 복사: 보정하지 않은 샘플도 `output_2`로 일괄 복사 가능
- 데이터셋 export: `create/output_2` 결과를 `create/dataset`으로 통합
- 학습 준비: `create/dataset`을 train/val로 나누고 `dataset.yaml` 생성
- YOLO 학습: `run.py train`에서 split 생성, label 정규화, 학습 실행

## 디렉터리 구조

```text
.
├─ config.py
├─ run.py
├─ Dockerfile
├─ docker-compose.yml
├─ requirements-docker.txt
├─ weights/
│  ├─ yolo26l-seg_finetuned_mix_2000.pt
│  └─ sam_vit_b_01ec64.pth
├─ create/
│  ├─ input/
│  │  └─ images/
│  ├─ output_1/
│  │  ├─ images/
│  │  ├─ masks/
│  │  └─ labels/
│  ├─ output_2/
│  │  ├─ images/
│  │  ├─ masks/
│  │  └─ labels/
│  ├─ dataset/
│  │  ├─ images/
│  │  ├─ masks/
│  │  └─ labels/
│  └─ training/
└─ pipeline/
```

## 준비

### 1. 모델 파일 배치

기본 설정은 [config.py](config.py)에 있습니다.

기본 필요 파일:

- YOLO segmentation weight: `weights/yolo26l-seg_finetuned_mix_2000.pt`
- SAM checkpoint: `weights/sam_vit_b_01ec64.pth`

파일명이 다르면 `config.py`의 `YOLO_WEIGHT`, `SAM_WEIGHT`를 수정하거나, segmentation 실행 시 `--weights`로 YOLO weight를 지정하세요.

### 2. 입력 이미지 이름

자동 세그멘테이션은 파일명 첫 토큰을 class id로 사용합니다.

권장 예시:

```text
0_1.png
0_2.png
1_1.png
2_15.jpg
```

`class_2_8.png`처럼 시작하는 파일은 다음 명령으로 `2_8.png` 형태로 바꿀 수 있습니다.

```bash
python run.py fix-names
```

## Docker 실행

이 프로젝트는 Docker Compose 기준으로 바로 Web GUI가 뜨도록 구성되어 있습니다. GPU 사용을 전제로 `docker-compose.yml`에 `gpus: all`이 설정되어 있습니다.

```bash
docker compose build
docker compose up annotation
```

브라우저에서 접속:

```text
http://localhost:7860
```

한 번만 실행하고 종료되는 CLI 명령은 다음 형태로 실행합니다.

```bash
docker compose run --rm annotation python run.py segment
```

GUI처럼 포트를 열어야 하는 명령은 다음처럼 실행합니다.

```bash
docker compose run --rm --service-ports annotation python run.py gui --host 0.0.0.0 --port 7860
```

## 로컬 실행

로컬 Python 환경에서 실행하려면 `requirements-docker.txt`의 패키지가 필요합니다. CUDA, PyTorch, `segment-anything`, Ultralytics 환경이 맞아야 하므로 보통은 Docker 실행을 권장합니다.

GUI 실행:

```bash
python run.py gui
```

인자 없이 `python run.py`만 실행해도 GUI가 시작됩니다.

## 파이프라인 명령

### 1. 입력 이미지 준비

이미지를 직접 넣는 위치:

```text
create/input/images/
```

또는 Web GUI의 input 탭에서 업로드할 수 있습니다.

### 2. 1차 자동 마스크 생성

```bash
python run.py segment
```

입출력 경로와 weight 지정:

```bash
python run.py segment --input create/input/images --output create/output_1 --weights yolo26l-seg_finetuned_mix_2000.pt
```

결과:

- `create/output_1/images/`: 마스크 overlay 이미지
- `create/output_1/masks/`: binary mask PNG
- `create/output_1/labels/`: YOLO-seg label TXT

참고: `--weights`는 `weights/` 안의 `.pt` 파일만 허용합니다.

### 3. GUI 보정

```bash
python run.py gui --host 0.0.0.0 --port 7860
```

GUI는 다음 데이터를 사용합니다.

- 원본 이미지: `create/input/images/`
- 초기 마스크: `create/output_1/masks/`
- 초기 라벨: `create/output_1/labels/`
- 저장 결과: `create/output_2/`

보정 저장 시:

- 이미지: `create/output_2/images/{stem}.png`
- 마스크: `create/output_2/masks/{stem}_edited.png`
- 라벨: `create/output_2/labels/{stem}.txt`

GUI의 remaining copy 기능은 아직 보정하지 않은 샘플을 `output_1`에서 `output_2`로 복사합니다.

### 4. Dataset export

```bash
python run.py export --bg paper
```

`--bg` 값은 파일명 앞에 prefix로 붙습니다.

예시:

```text
create/output_2/images/2_8.png
  -> create/dataset/images/paper_2_8.png
create/output_2/labels/2_8.txt
  -> create/dataset/labels/paper_2_8.txt
```

지원 옵션:

```bash
python run.py export --bg floor --src create/output_2 --dst create/dataset
```

결과:

- `create/dataset/images/`
- `create/dataset/masks/`
- `create/dataset/labels/`

### 5. Train/Val split 생성

```bash
python run.py build --num-classes 7 --val-ratio 0.2
```

결과:

- `create/training/images/train/`
- `create/training/images/val/`
- `create/training/labels/train/`
- `create/training/labels/val/`
- `create/training/dataset.yaml`

split은 `배경 prefix + class id` 그룹 기준으로 나눕니다. 즉 export 이후의 파일명은 `paper_2_8.png`처럼 첫 토큰이 배경, 두 번째 토큰이 class id인 형태가 안전합니다.

### 6. YOLO 학습

```bash
python run.py train \
  --weights yolo26l-seg.yaml \
  --name exp1 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --num-classes 7 \
  --val-ratio 0.2
```

`train` 명령은 학습 전에 내부적으로 `build`를 실행하고, `create/training/dataset.yaml`을 기본 데이터셋 설정으로 사용합니다.

주요 옵션:

- `--weights`: `weights/` 안의 `.pt`, `.yaml`, `.yml`
- `--name`: 학습 실험 이름
- `--epochs`, `--batch`, `--imgsz`
- `--num-classes`, `--val-ratio`
- `--project`: 결과 저장 위치, 기본값은 `weights/`
- `--augment`: `low`, `medium`, `high`
- `--optimizer`: `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, `auto`
- `--resume`, `--cache`, `--amp`, `--cos-lr`

## 기타 명령

### 클래스별 샘플 수 확인

```bash
python run.py count
python run.py count --dir create/dataset/images
```

### 작업 폴더 정리

```bash
python run.py clean --mode dataset
python run.py clean --mode input
python run.py clean --mode output1
python run.py clean --mode output2
python run.py clean --mode training
python run.py clean --mode all
```

주의: 지정한 작업 폴더 내부 파일이 삭제됩니다.

### stage 결과 샘플 수 줄이기

```bash
python run.py trim stage --dir create/output_1 --keep 200
```

### dataset 샘플 수 줄이기

```bash
python run.py trim dataset
```

`trim dataset`은 [config.py](config.py)의 `STACK_TRIM_KEEP_PER_BG` 값을 기준으로 배경별, 클래스별 샘플 수를 줄입니다.

## 설정

[config.py](config.py)에서 주요 경로와 기본값을 관리합니다.

환경 변수로 override 가능한 값:

- `ANNOTATION_PROJECT_ROOT`
- `ANNOTATION_WORKSPACE`
- `ANNOTATION_WEIGHTS_DIR`
- `ANNOTATION_WEB_HOST`
- `ANNOTATION_WEB_PORT`

주요 기본값:

- `WORKSPACE_ROOT`: `create/`
- `INPUT_IMAGES_DIR`: `create/input/images/`
- `OUTPUT1_DIR`: `create/output_1/`
- `OUTPUT2_DIR`: `create/output_2/`
- `DATASET_DIR`: `create/dataset/`
- `TRAINING_DIR`: `create/training/`
- `WEB_PORT`: `7860`

## 현재 코드 기준 참고사항

- `run.py`가 지원하는 명령은 `segment`, `gui`, `export`, `build`, `train`, `clean`, `count`, `fix-names`, `trim`입니다.
- `pipeline/bag_extract.py` 파일은 남아 있지만 현재 `run.py`에 `extract` 명령으로 연결되어 있지 않습니다.
- 현재 `config.py`에는 `BAG_DIR`, `INPUT_RGBD_DIR`, `DATASET_RGBD` 설정이 없습니다. 따라서 RealSense `.bag` 추출 흐름은 바로 실행 가능한 기본 파이프라인으로 보지 않는 것이 안전합니다.
- `export` 명령에는 `--mode copy/move` 옵션이 없습니다. 현재 구현은 copy 방식입니다.
- `build` 명령에도 `--mode` 옵션이 없습니다. 현재 구현은 copy 방식입니다.
- GUI는 SAM checkpoint를 시작 시 로드하므로 `weights/sam_vit_b_01ec64.pth`가 없으면 GUI의 SAM 기능 또는 앱 시작이 실패할 수 있습니다.
