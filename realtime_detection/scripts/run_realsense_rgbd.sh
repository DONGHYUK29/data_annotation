#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KERI_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROS_WORKSPACE="$KERI_ROOT/ros/workspace"

conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

source /opt/ros/jazzy/setup.bash
source "$ROS_WORKSPACE/install/setup.bash"

ros2 run detection_bridge realsense_rgbd_publisher_node --ros-args \
-p camera_width:=640 \
-p camera_height:=480 \
-p camera_fps:=30
