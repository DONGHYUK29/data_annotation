# Third-party Components

## Intel RealSense SDK

- Upstream: `https://github.com/IntelRealSense/librealsense.git`
- Used commit: `9a0dd70db1a2c180b69c6c257cd2ee6120505499`
- Description: `v2.57.7-2-g9a0dd70db`

SDK 소스와 `.git`, build 결과는 중복과 용량을 피하기 위해 포함하지 않았습니다.

## Segment Anything

- Upstream: `https://github.com/facebookresearch/segment-anything.git`
- Requirement commit: `dca509fe793f601edb92606367a655c15ac00fdf`
- Checkpoint used by the tools: `sam_vit_b_01ec64.pth` (not included)

## Ultralytics

YOLO 학습, 추론과 공식 segmentation validator에 사용했습니다. 당시 Python 환경의 Ultralytics 버전은 8.4.9였습니다.

각 구성요소의 저작권과 라이선스는 해당 upstream 프로젝트를 따릅니다.
