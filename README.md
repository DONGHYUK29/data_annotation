# 데이터 어노테이션 파이프라인

RealSense **`.bag`**에서 프레임 이미지를 뽑고, **YOLO + SAM**으로 1차 세그먼트·라벨을 만든 뒤, **GUI**로 마스크를 정제하여 YOLO 세그멘테이션 학습 데이터까지 만드는 도구 모음입니다.

## 파이프라인 흐름

1. **추출** — `bag/` 의 `.bag` → `create/input/` 이미지  
2. **1차 마스킹** — `create/input/` → `create/output_1/` (마스크, 라벨, 오버레이 이미지)  
3. **GUI 정제** — `create/input/` + `output_1` 마스크 → `create/output_2/` (최종 정제)  
4. **Export** — `output_2` → `create/dataset/` (배경 접두사 붙여 통합)  
5. **Train/Val 분할** — `dataset/` → `create/training/` + `dataset.yaml`  

모든 **경로·가중치 파일명**은 프로젝트 루트의 **`config.py`** 에서만 수정합니다.  
다른 PC에서는 `config.py`의 `PROJECT_ROOT`, `WORKSPACE_ROOT`, `BAG_DIR`, `WEIGHTS_DIR` 등을 맞추면 됩니다.

환경 변수로 덮어쓰기(선택):

| 변수 | 설명 |
|------|------|
| `ANNOTATION_PROJECT_ROOT` | 프로젝트 루트 |
| `ANNOTATION_WORKSPACE` | `create/` 에 해당하는 워크스페이스 루트 |
| `ANNOTATION_WEIGHTS_DIR` | `weights/` 디렉터리 |

---

## 사전 준비

### Python

- 로컬 가상환경 기준: **Python 3.10** 권장 (기존 venv와 동일 계열).

### 가중치 (`weights/`)

`config.py`에 지정한 파일명으로 두세요 (예시).

| 파일 | 용도 |
|------|------|
| `yolo26x.pt` | 1차 검출·세그( `YOLO_WEIGHT` ) |
| `sam_vit_b_01ec64.pth` | SAM (`SAM_WEIGHT`) |

가중치 파일명을 바꿨다면 **`config.py`의 `YOLO_WEIGHT`, `SAM_WEIGHT`만 같이 수정**하면 됩니다.

### 폴더 구조 (기본)

```
프로젝트루트/
  config.py
  run.py
  weights/
  bag/                 # .bag 파일 (config.BAG_DIR)
  create/
    input/             # 추출된 원본 프레임
    output_1/          # 1차 자동 마스킹
    output_2/          # GUI 정제 결과
    dataset/           # export 후 통합 데이터
    training/          # train/val 분할 결과
```

---

## 로컬 실행 (가상환경)

### 1) 의존성 설치

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

**PyTorch**는 [공식 안내](https://pytorch.org/get-started/locally/)에서 CUDA 버전에 맞는 명령으로 별도 설치하는 것을 권장합니다.

**Segment Anything**은 PyPI가 불안정할 수 있어 Git 설치를 권장합니다.

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

로컬 venv에 이미 있는 경우와 맞추려면 **CLIP**(Ultralytics 일부 기능)도 동일하게 설치할 수 있습니다.

```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

### 2) 통합 진입점 `run.py`

프로젝트 **루트**에서 실행합니다.

#### Bag → 이미지 (`extract`)

- `config.BAG_DIR` 아래 `{번호}.bag` 형식(예: `1.bag`, `2.bag`)을 읽습니다.
- `--start` ~ `--end` 범위의 번호만 처리합니다.
- 각 bag에서 균일 간격으로 `--count` 장을 `create/input/`에 저장합니다.

```bash
python run.py extract --start 1 --end 7 --count 300
```

선택:

- `--bag-dir 경로` — `config.BAG_DIR` 대신 사용  
- `--out 경로` — `config.IMAGES_RAW_DIR` 대신 출력  

#### 워크스페이스 비우기 (`clean`)

```bash
python run.py clean --mode all
```

`--mode`: `dataset` | `input` | `stage1` | `stage2` | `training` | `all`  
(config의 `WORKSPACE_ROOT` 아래 해당 하위 폴더만 비웁니다. 삭제에 유의하세요.)

#### 입력 파일명 정리 (`fix-names`)

`class_2_8` → `2_8` 처럼 `class_` 접두사 제거 (`create/input/` 기준).

```bash
python run.py fix-names
```

`--dir`으로 다른 폴더 지정 가능.

#### 1차 자동 마스킹 (`segment`)

YOLO + SAM으로 `create/input/` → `create/output_1/` (`masks`, `labels`, `images`).

```bash
python run.py segment
```

선택:

```bash
python run.py segment --input /path/to/images --output /path/to/stage1_root
```

**GPU**: CUDA가 보이면 자동으로 사용합니다.

#### GUI 정제 (`gui`)

1차 마스크를 불러와 브러시/지우개/SAM 포인트로 수정하고 `create/output_2/`에 저장합니다.

```bash
python run.py gui
```

- 원본 이미지: `create/input/`  
- 초기 마스크: `create/output_1/masks/` (또는 이미 정제한 `output_2`의 `*_edited.png`)  

종료 시 “수정하지 않은 이미지도 output_2로 이동할지” 묻는 동작은 기존과 동일합니다.

#### 정제본 → 통합 dataset (`export`)

`output_2`의 마스크·이미지·라벨을 `create/dataset/`으로 모으고, 파일명 앞에 배경 접두사를 붙입니다 (예: `paper_1_15.png`).

```bash
python run.py export --bg paper --mode copy
```

- `--bg` (필수): 환경 - `paper`, `floor`, `desk`, 작업 순서 - '4-1', '27' 등  
- `--mode`: `copy` 또는 `move`  

#### Train / Val 분할 (`build`)

`create/dataset/images` + `labels` 쌍만 사용해 `create/training/`에 나누고 `dataset.yaml`을 생성합니다.

```bash
python run.py build --num-classes 7 --val-ratio 0.2 --mode copy
```

- `--num-classes` (필수): YAML의 클래스 수  
- `--val-ratio`: 검증 비율 (기본값은 `config.DEFAULT_VAL_RATIO`)  
- 파일명 규칙: `{배경}_{class}_{번째}.png` 형태(예: `paper_1_15`) — `build_split`의 그룹핑에 사용  

#### 클래스별 개수 확인 (`count`)

`create/dataset/images` 기준으로 파일명에 포함된 클래스 번호를 세어 출력합니다.

```bash
python run.py count
```

`--dir`으로 다른 이미지 폴더 지정 가능.

#### 샘플 수 줄이기 (`trim`)

**stage** (output_1 / output_2 와 같은 구조: `labels/`, `masks/`, `images/`)

```bash
python run.py trim stage --dir create/output_1 --keep 200
```

**dataset** (`config.STACK_TRIM_KEEP_PER_BG` 기준으로 배경·클래스별 개수 자르기)

```bash
python run.py trim dataset
```

dataset용 `--dir`은 `pipeline/trim.py`의 인자로 전달됩니다 (기본 `create/dataset`).

#### 도움말

인자 없이 실행하면 사용 가능한 하위 명령 개요가 출력됩니다.

```bash
python run.py
```

---

## Docker 실행

로컬 `venv`와 최대한 동일하게 맞추기 위해 `requirements-docker.txt`는 **`pip freeze` 기준으로 고정 버전**을 생성해 Docker에 사용합니다.  
Dockerfile에서는 `torch/torchvision/torchaudio`만 `TORCH_INDEX`(기본: cu121)로 설치하고, 나머지는 `requirements-docker.txt`에서 그대로 설치합니다.

### 사전 요구

- [Docker](https://docs.docker.com/get-docker/) / Docker Compose V2  
- **GPU 사용 시**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)  
- Windows: Docker Desktop + WSL2 백엔드에서 GPU 지원 설정(환경마다 상이)을 확인하세요.

### 빌드

프로젝트 루트에서:

```bash
docker compose build
```

**CPU 전용** 이미지(느리지만 GPU 없이 테스트할 때):

```bash
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t data-annotation-sw:cpu .
```

`docker-compose.yml`의 `build.args.TORCH_INDEX`를 위 CPU URL로 바꾼 뒤 `docker compose build` 해도 됩니다.

### 실행 패턴

현재 디렉터리를 `/app`에 마운트하므로 **호스트에서 수정한 `config.py`·데이터·가중치**가 그대로 반영됩니다.

```bash
# 예: 1차 세그먼트
docker compose run --rm annotation python run.py segment

# 예: bag 추출
docker compose run --rm annotation python run.py extract --start 1 --end 5 --count 100
```

**GPU가 없는 Docker 호스트**에서는 `docker-compose.yml`의 `gpus: all` 줄을 제거하거나 주석 처리한 뒤, 가능하면 **CPU용**으로 다시 빌드하세요.

### GUI in Docker

GUI를 Docker에서 실제로 화면에 띄우려면 **X11 forwarding**이 필요합니다.

`docker-compose.yml`에 GUI용 서비스인 `annotation_gui`를 추가해 두었습니다.
아래로 실행하면 컨테이너의 GUI가 호스트 X 서버로 표시됩니다.

#### 1) X 서버 준비
- Linux/WSL2: 호스트에서 X 서버가 떠 있어야 합니다.
- Windows: VcXsrv/Xming 같은 X 서버를 먼저 실행하고, “TCP 연결 허용(접근제어 off)” 또는 동등한 설정을 해주세요.

#### 2) 실행
프로젝트 루트에서:

```bash
docker compose run --rm annotation_gui python run.py gui
```

`annotation_gui`는 기본적으로 `DISPLAY=${DISPLAY:-host.docker.internal:0}` 로 설정하고, `QT_QPA_PLATFORM=xcb`로 강제합니다.

### `.dockerignore`

`venv/`, `bag/`, `create/input` 등 대용량 경로는 이미지 빌드 컨텍스트에서 제외할 수 있습니다.  
데이터는 **볼륨 마운트(`.` → `/app`)** 로 주입하는 것이 정상입니다.

---

## Ultralytics 학습 예시 (참고)

정제된 `create/training/dataset.yaml`을 사용해 세그멘테이션 학습을 돌릴 수 있습니다. (경로·가중치는 본인 환경에 맞게 수정.)

```bash
yolo segment train data=create/training/dataset.yaml model=weights/yolo26x-seg.pt epochs=150 imgsz=640 batch=32 device=0
```

---

## 문제 해결

| 증상 | 점검 |
|------|------|
| `pyrealsense2` 오류 | Linux/Windows에 맞는 wheel 설치, Docker에서는 호스트와 동일 아키텍처인지 확인 |
| CUDA not available | `nvidia-smi`, Docker GPU 설정, `segment`만 컨테이너에서 CPU 빌드인지 확인 |
| SAM/YOLO 가중치 오류 | `config.py` 경로와 `weights/` 파일 존재 여부 |
| Export 시 이미지 없음 | GUI 저장 파일명: 마스크는 `*_edited.png`, 이미지는 `stem.png` 규칙과 `export`의 `clean_filename` 로직 확인 |

---

## 라이선스·서드파티

- [Segment Anything (Meta)](https://github.com/facebookresearch/segment-anything)  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- [Intel RealSense](https://github.com/IntelRealSense/librealsense)  
