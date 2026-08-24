# Model Manifest

최종 ROS 전달 모델 한 개는 Git LFS로 `models/deployment/best.pt`에 보존했습니다. 나머지 모델은 저장소에서 제외했으며, 아래 SHA-256은 동일 파일인지 확인하기 위한 값입니다.

## 저장소에 포함된 전달 모델

| 경로 | 바이트 | SHA-256 |
| --- | ---: | --- |
| `models/deployment/best.pt` | 63,514,823 | `d4722cccd5b7bda2734a01d36e8a0c782847a86cc4b08802c3716d5d980f1f35` |

원본은 `Keri_project/ros/delivery/weights/best.pt`였으며, `전기연배경증강.pt`와 `배경증강후.pt`는 동일 hash의 중복 파일이었습니다.

## 프로젝트 핵심 모델

| 원본 파일명 또는 역할 | 바이트 | SHA-256 |
| --- | ---: | --- |
| `weights_results/dh_best.pt` | 63,442,565 | `1d6bb9cf7dc780f92a973c22d1ebc2766ad6bdb107c0345b07ec6d9e6a1662d1` |
| `weights_results/L_mix_200.pt` | 63,447,685 | `86020fd4a88978187ada54c297cbf30098b35ac67e941cc87816b172867912e4` |
| `weights_results/full_L_200.pt` | 63,449,221 | `9b706326e40b4f3660c6156f3a308a05db2d1b141c35da2bad87f29c52e9df89` |
| `weights_results/aug_v1.pt` | 63,448,517 | `0d5c472fc232e41a83656886873a2d85119c3189d75eaa73a178fdfbbcc0801b` |
| `weights_results/aug_v2.pt` | 63,449,989 | `e5e7b1e7dcf7d37482e1058714e7e0a9f3fa031db23947bcf5a8a351b283cf46` |
| `weights_results/light_aug_joint.pt` | 63,447,301 | `8f48f94055c416e92ca0a90be198ea4c89bef77e72d8e9743fade6ca756230df` |
| `weights_results/light_aug_ft.pt` | 63,443,397 | `0603f703e2e286305f4b84d781151619f77400039bb37c59fbf642780c5815ad` |
| `weights_results/light_scale_pos_v1_ft.pt` | 63,440,645 | `a6b987d9fbefcff037d2da83f044ebacfc6718a48139120294d0f7d0b00044a8` |
| `weights_results/light_scale_pos_gt_v1_ft_conservative.pt` | 63,440,965 | `c0a83195ca63c4a888b2b0cdda85d5083fd55b71f30ace4bede3d1ff581375c0` |

## 배경 증강 비교 모델과 중복

| 논리 모델 | 원본 복사본 | SHA-256 |
| --- | --- | --- |
| 증강 전 | `증강전.pt`, `배경증강전.pt`, `best_0703.pt` | `1993b2c998406d2f251447c6c48ce390c4421478ffd756647226683afe13faa1` |
| 전기연 배경 증강 후 | `전기연배경증강.pt`, `배경증강후.pt`, ROS delivery `best.pt` | `d4722cccd5b7bda2734a01d36e8a0c782847a86cc4b08802c3716d5d980f1f35` |
| 세종대 배경 증강 | `세종대배경증강.pt` | `16d7f83bac7d789c80197a98ca483bb35ff46949dfbad125d0fe932b0cab2d1f` |

## 기반 모델

| 모델 | 바이트 | SHA-256 |
| --- | ---: | --- |
| `yolo26l-seg.pt` | 63,700,037 | `636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c` |
| `yolo26m-seg.pt` | 54,750,385 | `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5` |
| `yolo26x-seg.pt` | 142,129,861 | `92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3` |
| Docker `yolo11l.pt` | 51,387,343 | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` |

> `yolo26l-seg.pt`의 원본에는 동일 hash의 복사본이 세 군데 있었습니다. 모델을 별도 보관할 때는 같은 hash당 한 파일만 남기면 됩니다.

## 복원 확인

```bash
sha256sum /path/to/model.pt
```

표의 값과 일치해야 보존 당시 파일과 동일합니다. 실제 배포에 사용한 모델은 Git LFS 포인터로 관리되므로 새 clone에서는 `git lfs pull`을 실행해야 합니다.
