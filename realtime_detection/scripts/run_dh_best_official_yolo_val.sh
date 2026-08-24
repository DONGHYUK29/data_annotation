#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KERI_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="${1:-${YOLO_MODEL_PATH:-}}"
DATA_YAML="${2:-$KERI_ROOT/sample_data/rgbd/dataset.yaml}"
DEVICE="${3:-0}"
IMGSZ="${4:-640}"
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 /absolute/path/to/model.pt [dataset.yaml] [device] [imgsz]" >&2
  exit 2
fi

yolo segment val \
  model="${MODEL_PATH}" \
  data="${DATA_YAML}" \
  split=test \
  imgsz="${IMGSZ}" \
  device="${DEVICE}" \
  project="$KERI_ROOT/results/generated" \
  name="dh_best_test_integrated_rgb_official_val" \
  exist_ok=True \
  plots=True \
  save_json=True
