#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KERI_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_WORKSPACE="$KERI_ROOT/ros/workspace"

conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

MODEL_PATH="${1:-${YOLO_MODEL_PATH:-}}"
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 /absolute/path/to/model.pt" >&2
  echo "or set YOLO_MODEL_PATH" >&2
  exit 2
fi

set +u
source /opt/ros/jazzy/setup.bash
source "$ROS_WORKSPACE/install/setup.bash"
set -u

ros2 launch detection_bridge realsense_yolo.launch.py \
  model_path:="$MODEL_PATH" \
  show_window:=true \
  inference_device:=0 \
  use_depth_refine:=false
