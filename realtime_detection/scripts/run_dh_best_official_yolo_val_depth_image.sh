#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KERI_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_PACKAGE_ROOT="$KERI_ROOT/ros/workspace/src/detection_bridge"

MODEL_PATH="${1:-${YOLO_MODEL_PATH:-}}"
SOURCE_DATASET="${2:-$KERI_ROOT/sample_data/rgbd}"
DEPTH_DATASET="${3:-$KERI_ROOT/data/generated_depth_yolo}"
DEVICE="${4:-0}"
IMGSZ="${5:-640}"
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 /absolute/path/to/model.pt [source_dataset] [depth_dataset] [device] [imgsz]" >&2
  exit 2
fi

python3 "$ROS_PACKAGE_ROOT/tools/prepare_depth_yolo_dataset.py" \
  --source "${SOURCE_DATASET}" \
  --output "${DEPTH_DATASET}" \
  --overwrite

yolo segment val \
  model="${MODEL_PATH}" \
  data="${DEPTH_DATASET}/dataset.yaml" \
  split=test \
  imgsz="${IMGSZ}" \
  device="${DEVICE}" \
  project="$KERI_ROOT/results/generated" \
  name="dh_best_test_integrated_depth_image_official_val" \
  exist_ok=True \
  plots=True \
  save_json=True
