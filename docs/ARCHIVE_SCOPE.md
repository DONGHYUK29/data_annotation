# Archive Scope and Cleanup Boundary

## 이 Git 저장소에 보존한 것

- 직접 작성·수정한 Python, ROS 2, launch, message, Docker 소스
- 재실행에 필요한 핵심 설정과 사용법
- RGB-D/YOLO label 형식을 보여주는 샘플 한 건
- 대표 학습 곡선, 혼동행렬, PR curve와 bad-case CSV
- Git LFS로 추적한 최종 ROS 전달 모델 `models/deployment/best.pt`
- 모델 hash, 환경 버전, 기존 저장소와 외부 의존성 기록

## 의도적으로 제외한 것

- `data_annotation`의 input/output, dataset, training, test, overlay, mix 데이터
- RealSense RGB-D 원본 캡처와 배경 동영상
- Ultralytics 전체 `runs/`와 last/best checkpoint
- 최종 전달 모델을 제외한 `.pt`, `.pth`, Docker 내부 모델 파일
- ROS `build`, `install`, `log`와 사용자 runtime log
- Conda/Python 패키지 백업과 Ultralytics 사용자 설정
- librealsense checkout 및 빌드 결과
- 중복 전달본, 압축 파일, 동영상과 빈 legacy 디렉터리

## 삭제 전 검증 기준

1. 이 저장소가 GitHub에 push되어야 합니다.
2. 별도 경로에 다시 clone한 뒤 파일 수와 대용량 파일 검사를 통과해야 합니다.
3. Python 구문 검사와 ROS 패키지 빌드를 통과해야 합니다.
4. `models/deployment/best.pt`를 `MODEL_MANIFEST.md`의 hash와 대조해야 합니다.
5. 대체할 수 없는 RGB-D 원본을 별도 저장할지 최종 확인해야 합니다.

검증 전에는 `/home/prml513/Keri_project` 원본을 삭제하지 않습니다.
