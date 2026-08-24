# Preserved Results

전체 Ultralytics `runs/` 대신 비교에 필요한 대표 산출물만 남겼습니다.

## 학습 결과

| 실험 | 최고 epoch | mask mAP50 | mask mAP50-95 | box mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `light_aug_v2_ft_e20_lr1e3` | 13 | 0.99490 | 0.98930 | 0.99410 |
| `light_aug_v2_joint-2` | 25 | 0.99454 | 0.98872 | 0.99421 |
| `light_scale_pos_bg_v1_ft` | 2 | 0.99473 | 0.98832 | 0.99159 |
| `train-3` | 15 | 0.99488 | 0.98717 | 0.99340 |

값은 보존된 `results.csv`에서 mask mAP50-95가 가장 높은 행을 선택해 요약했습니다. 각 디렉터리에는 당시의 `args.yaml`, 전체 epoch CSV, 학습 곡선과 혼동행렬이 있습니다. `args.yaml`의 절대경로는 당시 실행 기록이므로 수정하지 않았습니다.

## 평가 결과

- `evaluation/rgb/`: 최종 RGB 평가의 PR 곡선, 혼동행렬과 예측 비교
- `evaluation/depth_image/`: depth image 변환 평가의 같은 산출물
- `evaluation/bad_case_report_all.csv`: `L_mix_200` bad-case 전체 보고서

전체 예측 이미지와 JSON, 모델 가중치 및 데이터셋은 용량 때문에 제외했습니다.
