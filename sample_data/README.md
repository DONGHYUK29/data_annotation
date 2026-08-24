# Sample Data

`rgbd/`는 전체 평가 데이터에서 한 건만 남긴 실행 확인용 샘플입니다.

```text
rgbd/
├── images/0_1.png      # RGB 640×480
├── depth/0_1.depth     # np.save로 저장된 uint16 depth, 640×480
├── labels/0_1.txt      # YOLO segmentation polygon
└── dataset.yaml
```

Depth scale은 기본 `0.001`m입니다. 이 샘플은 학습이나 성능 비교를 대표하지 않으며, 입출력 형식과 도구 동작을 확인하기 위한 것입니다.
