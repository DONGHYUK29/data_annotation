# 데이터 어노테이션 파이프라인 README

RealSense **`.bag`** 파일에서 프레임 이미지를 추출하고, **YOLO**으로 1차 마스킹 및 세그먼테이션을 수행한 뒤, **GUI**에서 마스크를 정제하여 최종적으로 **YOLO 세그멘테이션 학습 데이터셋**까지 생성하는 파이프라인입니다.

---

## 1. 파이프라인 흐름

전체 작업 순서는 다음과 같습니다.

1. **Extract**  
   `bag/` 폴더의 `.bag` 파일에서 프레임 이미지를 추출하여 `create/input/`에 저장


2. **Segment**  
   `create/input/` 이미지를 대상으로 YOLO 기반 1차 자동 마스킹 수행  
   결과는 `create/output_1/`에 저장

   저장 항목: 오버레이 이미지, 클래스/bbox/마스크 정보 txt, 마스크 PNG


3. **GUI 정제**  
   자동 생성된 마스크를 GUI에서 수정 및 보정  
   결과는 `create/output_2/`에 저장
   저장 항목: 오버레이 이미지, 수정된 클래스/bbox/마스크 정보 txt, 수정된 마스크 PNG


4. **Export**  
   정제된 결과를 `create/dataset/`으로 통합


5. **Build**  
   `dataset`을 train / val로 분할하고 `dataset.yaml` 생성

즉, 전체 파이프라인은 아래와 같이 정리할 수 있습니다.

```text
.bag 파일 → 이미지 추출 → 1차 자동 마스킹 → GUI 정제 → dataset 통합 → train/val 분할
```

---

## 2. 사전 준비

본 프로젝트는 Docker 기반 실행을 기준으로 사용합니다.

### 2.1 필수 설치 항목

#### Windows (NVIDIA GPU 사용 기준)
- NVIDIA 그래픽 드라이버 설치
- WSL2 설치 및 활성화
- WSL2용 Linux 배포판 설치 (recommend: Ubuntu)
- WSL2 업데이트 (`wsl --update`)
- Docker Desktop 설치
- Docker Desktop에서 WSL2 backend 활성화

※ Docker Desktop에는 Docker Engine, Docker CLI, Docker Compose가 포함되므로 별도 Compose 설치는 일반적으로 필요하지 않음

#### macOS
- Docker Desktop 설치
- Docker Desktop에는 Docker Engine, Docker CLI, Docker Compose가 포함되므로 별도 Compose 설치는 일반적으로 필요하지 않음
- 단, 현재 Dockerfile은 NVIDIA CUDA 기반 이미지이므로 기본 설정 그대로는 GPU 가속 대상이 아님

#### Linux (NVIDIA GPU 사용 기준)
- Docker Engine 설치
- Docker Compose plugin 설치
- NVIDIA 드라이버 설치
- NVIDIA Container Toolkit 설치

### 2.2 프로젝트 파일 준비
- 저장소 clone
- `weights/` 폴더에 YOLO/SAM 가중치 배치
- `bag/` 폴더 또는 입력 이미지 준비
- `config.py`에서 경로 및 가중치 파일명 확인

### 2.3 GUI 실행 방식
- GUI는 웹 기반으로 실행됨
- 컨테이너 실행 후 브라우저에서 접속하여 사용
- 기본 포트는 7860
- http://localhost:7860/

### 2.4 기본 폴더 구조

```text
프로젝트루트/
  config.py
  run.py
  weights/
  bag/
  create/
    input/
    output_1/
    output_2/
    dataset/
    training/
```

---

## 3. 파일 설명

### 3.1 `config.py`

`config.py`는 프로젝트 전반에서 사용하는 **디렉토리 경로와 가중치 파일 경로를 관리하는 설정 파일**입니다.

다른 PC에서 프로젝트를 실행할 때는 가장 먼저 이 파일을 확인해야 합니다.  
주요 설정 대상은 다음과 같습니다.

- `PROJECT_ROOT`
- `WORKSPACE_ROOT`
- `BAG_DIR`
- `WEIGHTS_DIR`
- `YOLO_WEIGHT`
- `SAM_WEIGHT`

필요하면 환경 변수로 일부 경로를 덮어쓸 수 있습니다.

- `ANNOTATION_PROJECT_ROOT`
- `ANNOTATION_WORKSPACE`
- `ANNOTATION_WEIGHTS_DIR`

### 3.2 `run.py`

`run.py`는 이 프로젝트의 **통합 실행 진입점**입니다.  
각 단계는 `run.py`의 하위 명령 형태로 실행합니다.

명령어는 다음과 같습니다.

주요 명령어
- `extract` : `.bag` 파일에서 이미지 추출
- `segment` : 1차 자동 마스킹 수행
- `gui` : GUI 기반 마스크 정제
- `export` : 정제 결과를 dataset 형식으로 통합


기타 명령어
- `build` : train / val 분할 및 `dataset.yaml` 생성
- `count` : 클래스별 샘플 수 확인
- `clean` : 작업 폴더 삭제
- `trim` : 샘플 수 줄이기
- `fix-names` : 입력 파일명 정리

---

## 4. Docker 실행 전체 흐름

아래 순서대로 진행하면 전체 파이프라인을 실행할 수 있습니다.

### 4.1 Docker 이미지 빌드

프로젝트 루트에서 먼저 이미지를 빌드합니다. (최초 1회만 빌드, 이후에는 Docker Desktop -> Containers -> data_annotation_sw -> Actions -> start)

```bash
docker compose build
```

GPU가 없는 환경에서는 CPU 전용 이미지로 별도 빌드할 수도 있습니다.

```bash
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t data-annotation-sw:cpu .
```

---

### 4.2 `.bag` 파일에서 이미지 추출

`bag/` 폴더의 `.bag` 파일에서 프레임 이미지를 뽑아 `create/input/`에 저장합니다.

```bash
docker compose run --rm annotation python run.py extract --start 0 --end 9 --count 100
```

#### 주요 옵션
- `--start` : 시작 bag 번호
- `--end` : 끝 bag 번호
- `--count` : 각 bag에서 추출할 이미지 수
- `--bag-dir` : bag 폴더 직접 지정
- `--out` : 추출 이미지 저장 폴더 직접 지정

---

### 4.3 1차 자동 마스킹

YOLO + SAM으로 자동 세그멘테이션을 수행합니다.  
결과는 `create/output_1/`에 저장됩니다.

```bash
docker compose run --rm annotation python run.py segment
```

필요하면 입력 / 출력 폴더를 직접 지정할 수 있습니다.

```bash
docker compose run --rm annotation python run.py segment --input /path/to/images --output /path/to/stage1_root
```

---

### 4.4 GUI 정제

자동 생성된 마스크를 GUI에서 수정합니다.  
브러시, 지우개, SAM 포인트 등을 사용해 마스크를 직접 보정할 수 있습니다.

```bash
docker compose run --rm --service-ports annotation python run.py gui --host 0.0.0.0 --port 7860
```
http://localhost:7860/
   <- 해당 주소에서 GUI 프로그램 실행

기본적으로 다음 데이터를 사용합니다.

- 원본 이미지: `create/input/`
- 초기 마스크: `create/output_1/masks/`
- 저장 결과: `create/output_2/`

---

### 4.5 Export

GUI에서 정제한 결과를 최종 dataset 구조로 모읍니다.  
이때 파일명 앞에 배경 접두사를 붙여 통합합니다.

```bash
docker compose run --rm annotation python run.py export --bg paper --mode copy
```

#### 주요 옵션
- `--bg` : 배경 이름 또는 작업 구분자
- `--mode` : `copy` 또는 `move`

예시 파일명:
```text
paper_1_15.png
```

---

## 5. 기타 기능

아래 기능들은 전체 GUI 흐름과 직접 연결되지는 않지만, 작업 정리나 데이터 관리에 유용합니다.


### 5.1 Train / Val 분할

최종 dataset을 학습용 / 검증용으로 나누고 `dataset.yaml`을 생성합니다.

```bash
docker compose run --rm annotation python run.py build --num-classes 7 --val-ratio 0.2 --mode copy
```

#### 주요 옵션
- `--num-classes` : 클래스 수
- `--val-ratio` : validation 비율
- `--mode` : `copy` 또는 `move`

---

### 5.2 전체 삭제 / 작업 폴더 비우기

워크스페이스 하위 폴더를 정리할 때 사용합니다.

```bash
python run.py clean --mode all
```

사용 가능한 모드:
- `dataset`
- `input`
- `stage1`
- `stage2`
- `training`
- `all`

> 주의: 해당 폴더 내용이 삭제되므로 사용 시 주의해야 합니다.

---

### 5.3 입력 파일명 정리

예를 들어 `class_2_8` 같은 이름에서 `class_` 접두사를 제거할 때 사용합니다.

```bash
python run.py fix-names
```

필요하면 `--dir` 옵션으로 다른 폴더를 지정할 수 있습니다.

---

### 5.4 클래스별 개수 확인

현재 dataset 내 클래스별 샘플 수를 확인합니다.

```bash
python run.py count
```

필요하면 `--dir` 옵션으로 다른 경로를 지정할 수 있습니다.

---

### 5.5 샘플 수 줄이기

테스트용으로 데이터 수를 줄이거나, 배경/클래스별 개수를 조절할 때 사용합니다.

#### stage 기준 줄이기
```bash
python run.py trim stage --dir create/output_1 --keep 200
```

#### dataset 기준 줄이기
```bash
python run.py trim dataset
```

---

## 6. 권장 실행 순서

실제로는 아래 순서만 따라가면 됩니다.

1. `config.py`에서 경로 및 가중치 파일 위치 확인
2. Docker / Docker Compose 준비
3. Docker 이미지 빌드
4. `extract` 실행
5. `segment` 실행
6. `gui` 실행
7. `export` 실행
8. `build` 실행

---

## 7. 참고 사항

- 컨테이너는 현재 디렉터리를 `/app`에 마운트하므로, 호스트에서 수정한 `config.py`, 데이터, 가중치가 그대로 반영됩니다.
- GPU가 없는 환경에서는 `docker-compose.yml`의 GPU 관련 설정을 제거하거나 CPU 전용 이미지로 빌드해야 합니다.
- 문제가 발생하면 우선 `config.py`의 경로 설정과 `weights/` 파일 존재 여부를 먼저 확인하는 것이 좋습니다.
