# Source Provenance

2026-08-24에 DGX 홈 디렉터리에 흩어져 있던 로봇팔 과제 자료를 `/home/prml513/Keri_project`로 통합한 뒤, 이 저장소를 소스 중심 snapshot으로 별도 구성했습니다.

## 기존 Git 저장소

| 작업공간 | 원격 | 당시 HEAD | 상태 |
| --- | --- | --- | --- |
| 초기 annotation pipeline | `https://github.com/sangwoo-eom/data_annotation.git` | `7718f43039ff321ce8173ae93f75f92b6f2a39e2` | 수정 및 미추적 파일 존재 |
| 확장 실험 작업공간 | `https://github.com/DONGHYUK29/data_annotation.git` | `f1725d66776b501593c7a925b4b7cf8bee00e33e` | 대규모 데이터와 미추적 실험 코드 존재 |
| librealsense | `https://github.com/IntelRealSense/librealsense.git` | `9a0dd70db1a2c180b69c6c257cd2ee6120505499` | clean |

이 snapshot은 기존 저장소의 중첩 `.git` 기록을 합친 것이 아닙니다. 당시 working tree에서 실제 사용하던 최신 소스만 선별했으므로, 위 두 annotation 원격 저장소보다 현재 코드가 더 최신일 수 있습니다.

## 원본 위치 대응

| 과거 위치 | 통합 원본 위치 |
| --- | --- |
| `~/project/data_annotation` | `Keri_project/data_annotation/pipeline` |
| `~/project/data_annotation_dh` | `Keri_project/data_annotation/experiments_dh` |
| `~/ros2_ws` | `Keri_project/ros/workspace` |
| `~/project/librealsense` | `Keri_project/ros/librealsense` |
| `~/detection_bridge_delivery` | `Keri_project/ros/delivery` |
| 홈의 `run_*` 스크립트 | `Keri_project/realtime_detection/scripts` |

캡스톤, BAMTI, IoT 프로젝트는 이 저장소에 포함하지 않았습니다.
