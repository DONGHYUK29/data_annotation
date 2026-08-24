# Detection Bridge ROS 2

RealSense RGB, aligned depth, and camera info를 입력받아 YOLO Segmentation 추론을 수행하고, 결과를 ROS 2 토픽으로 publish하는 패키지입니다.

구성은 두 개의 패키지로 나뉩니다.

- `detection_bridge`: 실행 노드 패키지
- `detection_bridge_msgs`: 사용자 정의 메시지 패키지

## 1. 시스템 개요

패키지는 다음 흐름으로 동작합니다.

1. RGB와 Depth를 개별 raw 수신 카운터와 함께 입력 토픽에서 수신합니다.
2. CameraInfo는 별도 subscription으로 수신하여 최신 intrinsics를 보관합니다.
3. RGB와 Depth만 synchronized callback으로 묶어 YOLO Segmentation inference를 수행합니다.
4. 객체별 bbox, mask, center, depth, XYZ, instance mask를 계산합니다.
5. `DetectionArray`를 publish한 후 선택적으로 시각화를 수행합니다.
6. OpenCV 화면에는 작은 반투명 패널로 FPS와 latency를 표시합니다.

`yolo_realsense_node`와 `yolo_realsense_depth_refine_node`는 같은 베이스 구현을 공유하며, depth-based mask refinement 기본값만 다릅니다.

## 2. 지원 환경

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12
- OpenCV
- cv_bridge
- message_filters
- ultralytics
- numpy
- pyrealsense2

## 3. 필수 의존성

ROS 의존성:

- `rclpy`
- `sensor_msgs`
- `std_msgs`
- `cv_bridge`
- `message_filters`
- `launch`
- `launch_ros`
- `detection_bridge_msgs`

Python 의존성:

- `ultralytics`
- `numpy`
- `opencv-python` 또는 ROS 환경의 OpenCV 바인딩
- `pyrealsense2` for `realsense_rgbd_publisher_node`

## 4. 빌드 방법

```bash
cd /path/to/Keri_project/ros/workspace
source /opt/ros/jazzy/setup.bash
colcon --log-base log_relocated build \
  --base-paths src/detection_bridge_msgs src/detection_bridge \
  --build-base build_relocated \
  --install-base install_relocated \
  --symlink-install \
  --packages-select detection_bridge_msgs detection_bridge \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
source install_relocated/setup.bash
```

If your shell is inside a conda environment, keep the system Python override above so ROSIDL can find `empy`.

## 5. 실행 방법

### RealSense + YOLO를 함께 실행

```bash
ros2 launch detection_bridge realsense_yolo.launch.py model_path:=/path/to/your_segmentation.pt
```

### YOLO만 실행

이미 RealSense ROS 2 노드가 다른 터미널에서 실행 중일 때 사용합니다.

```bash
ros2 launch detection_bridge yolo_only.launch.py model_path:=/path/to/your_segmentation.pt
```

### depth refinement 사용

```bash
ros2 launch detection_bridge realsense_yolo.launch.py \
  model_path:=/path/to/your_segmentation.pt \
  use_depth_refine:=true
```

기존 shell script도 그대로 사용할 수 있습니다.

- `~/run_realsense_rgbd.sh`
- `~/run_realsense.sh`
- `~/run_yolo_ros.sh`
- `~/run_yolo_ros_depth.sh`

## 6. 가중치 경로 변경

`model_path` ROS parameter로 YOLO Segmentation `.pt` 파일을 전달합니다.

우선순위는 다음과 같습니다.

1. launch argument `model_path`
2. launch YAML의 `model_path`
3. 환경변수 `YOLO_MODEL_PATH`

`model_path`가 비어 있으면 노드는 시작 시 오류를 발생시킵니다.

## 7. 입력 토픽

기본 입력 토픽은 launch YAML에서 설정합니다.

- RGB Image: `/camera/camera/color/image_raw`
- Aligned Depth Image: `/camera/camera/aligned_depth_to_color/image_raw`
- CameraInfo: `/camera/camera/color/camera_info`

## 8. 출력 토픽

- Core detection result: `/detection_result`
- Debug JSON: `/detection_result_json`
- Instance mask image: `/instance_mask/image`
- Annotated image: `/annotated_image`

`result_topic`, `result_json_topic`, `instance_mask_topic`, `annotated_image_topic` 모두 ROS parameter로 변경할 수 있습니다.

## 9. 메시지 필드

### Detection

기존 필드는 유지됩니다.

- `instance_id`
- `class_id`
- `class_name`
- `confidence`
- `final_confidence`
- `depth_quality`
- `bbox_xyxy`
- `center_xy`
- `xyz_m`
- `mask_area`
- `refined_mask_area`
- `depth_refine_applied`
- `depth_refine_reason`
- `depth_refine_removed_pixels`
- `depth_refine_area_ratio`

추가 필드:

- `bbox_center_xy`: bbox 중심
- `mask_center_xy`: segmentation mask centroid
- `depth_valid`: 대표 depth 유효 여부
- `depth_m`: 대표 depth

`center_xy`는 기존과 호환을 위해 유지되며, 현재 선택된 대표 중심을 뜻합니다. `use_refined_mask_center=true`면 mask centroid, 아니면 bbox center를 사용합니다.

### DetectionArray

기존 필드는 유지됩니다.

- `input_fps`
- `processed_fps`
- `processing_fps`
- `processing_latency_ms`
- `output_fps`
- `depth_refine_enabled`
- `detections`

추가 필드:

- `rgb_input_fps`
- `depth_input_fps`
- `synced_input_fps`
- `publish_fps`
- `end_to_end_latency_ms`
- `latency_equivalent_fps`
- `preprocess_latency_ms`
- `inference_latency_ms`
- `postprocess_latency_ms`
- `ros_message_latency_ms`

호환성을 위해 legacy 필드도 계속 채웁니다.

## 10. FPS 정의

- `rgb_input_fps`: RGB 메시지 수신 개수 / 실제 경과시간
- `depth_input_fps`: Depth 메시지 수신 개수 / 실제 경과시간
- `synced_input_fps`: RGB와 Depth가 synchronized callback으로 들어온 개수 / 실제 경과시간
- `publish_fps`: `DetectionArray`가 실제 publish된 개수 / 실제 경과시간
- `latency_equivalent_fps`: `1.0 / (t_pub - t_recv)`

`input_fps`, `processed_fps`, `processing_fps`, `output_fps`는 호환성용 legacy 필드로 유지되며, 새 필드와 함께 채워집니다.

## 11. End-to-End Latency

최종 latency는 다음으로 정의합니다.

- `t_recv`: synchronized RGBD callback 진입 직후의 `time.perf_counter()`
- `t_publish_ready`: `DetectionArray` 생성이 끝난 뒤, publish 호출 직전의 `time.perf_counter()`

계산식:

```text
end_to_end_latency_ms = (t_publish_ready - t_recv) * 1000.0
latency_equivalent_fps = 1.0 / (t_publish_ready - t_recv)
```

이 latency에는 다음이 포함됩니다.

- callback 진입
- ROS Image to OpenCV/NumPy 변환
- 전처리
- YOLO inference
- mask, bbox, confidence, depth, center, XYZ 계산
- Detection 및 DetectionArray 생성
- instance mask image 생성
- DetectionArray publish 직전

`publish()` 호출 자체의 소요시간은 `pub_call`로 별도 측정합니다. 로그의
`latency_return`은 callback 진입부터 `publish()` 반환까지의 참고값입니다.

다음은 포함하지 않습니다.

- segmentation overlay
- contour drawing
- bounding box drawing
- text drawing
- `cv2.imshow()`
- `cv2.waitKey()`

## 12. 화면 표시 항목

OpenCV 화면에는 다음이 표시됩니다.

- RGB image
- segmentation mask overlay
- segmentation contour
- bounding box
- class name
- confidence
- depth
- bbox center point
- optional mask center point
- RGB input FPS
- Depth input FPS
- synchronized callback FPS
- DetectionArray publish FPS
- end-to-end latency
- optional inference latency

성능 패널은 영상 우측 상단에 작은 글씨와 반투명 배경으로 표시하여 탐지 화면을
가리는 영역을 최소화합니다.

기본 설정은 bbox center를 눈에 잘 띄게 표시하고, mask center는 `draw_mask_center:=true`일 때 추가로 표시합니다.

## 13. Depth가 유효하지 않을 때

Depth가 유효하지 않아도 detection과 segmentation 결과는 유지합니다.

- detection은 publish됩니다.
- `depth_valid=false`로 표시됩니다.
- `depth_m`와 `xyz_m`는 안전한 기본값인 `0.0`을 사용합니다.
- 화면에는 `Z:N/A` 또는 `X:N/A Y:N/A Z:N/A`로 표시됩니다.

대표 depth 계산은 다음 순서를 유지합니다.

1. refined mask core 영역의 valid depth median
2. 중심 주변 valid depth median fallback

## 14. 종료 및 재실행

- OpenCV 창은 `ESC`로 종료할 수 있습니다.
- 터미널에서 `Ctrl+C`로 종료할 수 있습니다.
- 종료 시 `cv2.destroyAllWindows()`와 RealSense pipeline stop이 호출됩니다.
- `rclpy.shutdown()`은 중복 호출되지 않도록 처리합니다.
- launch로 시작한 경우 자식 노드도 함께 종료됩니다.

재실행이 안 될 경우 먼저 이전 프로세스가 남아 있는지 확인합니다.

```bash
ps aux | grep -E 'realsense|yolo|detection_bridge' | grep -v grep
```

## 15. 문제 해결

- 모델 로드 실패: `model_path`가 올바른 `.pt` 파일인지 확인합니다.
- 토픽 수신 실패: RGB, Depth, CameraInfo 토픽 이름과 QoS를 확인합니다.
- `rgb/depth>0`, `synced=0`: RGB/Depth stamp 차이를 확인하고 `sync_slop_sec`를 조정합니다.
- stamp 차이 확인: `debug_stamp_delta:=true`와 ROS debug log level을 사용합니다.
- bundled `realsense_rgbd_publisher_node`는 `image_qos_reliability:=reliable`을 사용합니다.
- 외부 카메라 드라이버가 best-effort publisher라면 `image_qos_reliability:=best_effort`로 실행합니다.
- depth가 항상 N/A: aligned depth 토픽이 실제로 color와 정렬되어 있는지 확인합니다.
- 화면이 안 뜸: `show_window:=true`인지 확인하고, GUI 세션인지 확인합니다.
- instance mask가 안 나옴: `publish_instance_mask:=true`인지 확인합니다.
- publish FPS가 0에 가까움: publish throttling(`publish_every_sec`)과 YOLO 모델 속도를 확인합니다.

## 16. 참고

`src/yolo_ros`는 별도 패키지로 존재하지만, 현재 `detection_bridge` 실행 경로에서는 참조하지 않습니다.
