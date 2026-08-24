# Experiment Record

## 데이터 수집과 라벨링

- RealSense RGB 배경 영상을 640×480, 30 FPS로 수집했습니다.
- 15초 RGB-D 수집 GUI는 약 0.25초 간격으로 RGB와 aligned depth pair를 저장했습니다.
- YOLO bbox를 prompt로 사용해 SAM `vit_b` mask를 만들었습니다.
- PySide6 GUI와 brush/SAM editor로 mask와 YOLO polygon을 수정했습니다.

## 증강

주요 실험 축은 다음과 같습니다.

- 배경 색상·난이도 변경: 객체 mask는 유지하고 배경만 변형
- 조명 변경: 밝기, 대비, gamma, CLAHE, shadow, highlight
- 위치·스케일 변경: 객체를 mask로 crop해 desk/paper/floor 배경에 합성
- 원본과 증강 데이터를 joint training하거나 기존 모델을 fine-tuning

관련 구현은 `data_annotation/tools/augmentation/`에 있습니다.

## 평가

- Ultralytics 공식 YOLO segmentation validator
- RGB 이미지 평가와 depth-to-image 평가 비교
- depth mask refinement 전후 공식 mask IoU/AP 계산
- 누락, class mismatch, 낮은 mask IoU, 낮은 confidence, false positive 기반 bad-case 점수화

대표 수치와 그래프는 `results/`에 있습니다.

## ROS 통합

최종 노드는 RGB와 depth를 approximate synchronization하고 다음을 계산했습니다.

- segmentation mask와 bbox
- bbox 또는 mask 중심
- 대표 depth와 카메라 intrinsics 기반 XYZ
- depth 경계 기반 mask refinement
- 입력/publish FPS와 전처리·추론·후처리·메시지 latency

출력은 `detection_bridge_msgs/DetectionArray`로 전달했습니다.
