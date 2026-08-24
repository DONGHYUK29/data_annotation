# Reproduction Environment

## 당시 DGX 환경

| 항목 | 값 |
| --- | --- |
| OS | Ubuntu 24.04.3 LTS |
| Kernel | `6.14.0-1015-nvidia` |
| GPU | NVIDIA GB10 |
| NVIDIA driver | `580.95.05` |
| CUDA toolkit | 13.0 |
| ROS | ROS 2 Jazzy |
| Conda environment | `robot` |
| NumPy | 2.4.1 |
| OpenCV Python | 4.10.0 |
| PyTorch | `2.11.0.dev20260104+cu130` |
| Torchvision | `0.25.0.dev20260104+cu130` |
| Ultralytics | 8.4.9 |
| PySide6 | 6.10.2 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.1 |

PyTorch는 당시 개발/nightly 빌드였으므로 다른 장비에서는 GPU와 CUDA에 맞는 공식 빌드를 먼저 설치하는 편이 안전합니다. `requirements.txt`는 코드가 직접 사용하는 핵심 패키지만 정리한 재구성용 목록입니다.

## ROS 의존성

ROS 패키지는 다음 기능을 사용합니다.

- `rclpy`, `sensor_msgs`, `std_msgs`
- `cv_bridge`, `message_filters`
- `launch`, `launch_ros`, `ament_index_python`
- `rosidl_default_generators`, `rosidl_default_runtime`
- Python의 `numpy`, `opencv`, `ultralytics`, `pyrealsense2`

## 하드웨어 전제

- Intel RealSense RGB-D 카메라
- CUDA 사용 시 NVIDIA GPU와 호환 드라이버
- GUI 도구 사용 시 X11/Wayland 디스플레이

실제 모델과 전체 데이터셋은 별도 복원해야 합니다. 파일 식별값은 `MODEL_MANIFEST.md`를 참고합니다.
