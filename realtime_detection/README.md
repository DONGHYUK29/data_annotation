# Real-time Detection

## 단독 Python 실행

DGX headless 출력:

```bash
python realtime_detection/yolo_realtime_dgx_headless.py \
  --model /path/to/segmentation.pt
```

OpenCV 화면과 TCP JSON 전송:

```bash
python realtime_detection/yolo_realtime.py \
  --model /path/to/segmentation.pt \
  --socket-host 127.0.0.1 \
  --socket-port 5000
```

두 스크립트 모두 RealSense RGB 640×480, depth 640×480, 30 FPS를 사용하며 bbox 중심의 depth를 intrinsics로 역투영해 XYZ를 계산합니다.

## 실행 스크립트

| 스크립트 | 역할 |
| --- | --- |
| `run_realsense.sh` | RGB 배경 영상 수집 GUI |
| `run_realsense_rgbd.sh` | ROS 2 RGB-D publisher |
| `run_yolo_ros.sh` | RGB 기반 ROS 탐지 |
| `run_yolo_ros_depth.sh` | depth refinement 적용 ROS 탐지 |
| `run_dh_best_official_yolo_val.sh` | Ultralytics 공식 segmentation 평가 |
| `run_dh_best_official_yolo_val_depth_image.sh` | depth를 영상으로 변환한 뒤 평가 |

모델 인자는 첫 번째 위치 인자 또는 `YOLO_MODEL_PATH` 환경변수로 전달합니다.

```bash
realtime_detection/scripts/run_yolo_ros_depth.sh /path/to/model.pt
```

평가 스크립트의 기본 데이터는 `sample_data/rgbd`입니다. 전체 평가를 재현하려면 같은 `images/`, `depth/`, `labels/` 구조의 원본 데이터셋 경로를 추가 인자로 지정합니다.
