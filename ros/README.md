# ROS 2 Detection Bridge

RealSense RGB와 aligned depth를 동기화하고 YOLO Segmentation을 수행한 뒤 객체별 bbox, mask, depth, XYZ 및 latency를 `DetectionArray`로 publish합니다.

## 환경

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12 계열 시스템 Python
- RealSense RGB-D 카메라
- Ultralytics YOLO Segmentation

## 패키지

- `detection_bridge_msgs`: `Detection.msg`, `DetectionArray.msg`
- `detection_bridge`: RealSense publisher, YOLO/depth-refinement 노드, launch와 평가 도구

## 빌드

```bash
cd ros/workspace
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select detection_bridge_msgs detection_bridge \
  --cmake-args \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

Conda 환경에서는 ROSIDL의 Python 패키지 충돌을 피하기 위해 위 시스템 Python 지정을 유지합니다.

## 실행

```bash
ros2 launch detection_bridge realsense_yolo.launch.py \
  model_path:=/path/to/segmentation.pt \
  use_depth_refine:=true
```

주요 출력 토픽:

- `/detection_result`: 사용자 정의 탐지 메시지
- `/detection_result_json`: 선택적 JSON 디버그 출력
- `/instance_mask/image`: 선택적 instance mask
- `/annotated_image`: 선택적 시각화 영상

세부 파라미터와 메시지 필드는 [`workspace/src/detection_bridge/README.md`](workspace/src/detection_bridge/README.md)에 정리되어 있습니다.

## RealSense SDK

기존 DGX에서는 Intel `librealsense`의 `9a0dd70db1a2c180b69c6c257cd2ee6120505499` 커밋(`v2.57.7-2-g9a0dd70db`)을 사용했습니다. SDK 전체 checkout과 빌드 산출물은 이 저장소에서 제외했습니다.
