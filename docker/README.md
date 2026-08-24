# Docker Benchmark Environment

`yolo11_pytorch/`는 `nvcr.io/nvidia/pytorch:24.01-py3` 기반의 초기 YOLO11 GPU FPS 측정 환경입니다.

```bash
docker build -t keri-yolo11-benchmark docker/yolo11_pytorch
docker run --rm --gpus all \
  -v "$PWD/docker/yolo11_pytorch:/workspace" \
  -v "$PWD/weights:/workspace/weights:ro" \
  -e YOLO_WEIGHTS_DIR=/workspace/weights \
  keri-yolo11-benchmark \
  python yolo_fps_square.py
```

Dockerfile에는 당시 OpenCV 충돌을 피하기 위해 system OpenCV 제거 후 `opencv-python-headless==4.8.1.78`을 재설치한 과정이 남아 있습니다. 모델 가중치와 Docker 이미지는 포함하지 않았습니다.
