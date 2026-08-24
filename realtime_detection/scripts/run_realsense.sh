#!/usr/bin/env bash
set -e

# ============================================================
# RealSense 배경 영상 촬영 실행 스크립트
# 사용법:
#   ./run_realsense.sh
#
# 저장 위치:
#   Keri_project/data/background_videos/session_YYYYMMDD_HHMMSS/
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KERI_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$KERI_ROOT"
SCRIPT_PATH="$KERI_ROOT/data_annotation/tools/capture/record_realsense_gui.py"

OUT_ROOT="$KERI_ROOT/data/background_videos"
PREFIX="lab_bg"

WIDTH=640
HEIGHT=480
FPS=30

# DGX 로컬 모니터에서 실행할 때 DISPLAY가 비어 있으면 기본값 설정
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:0
fi

echo "============================================================"
echo "RealSense Background Recorder"
echo "PROJECT_ROOT : $PROJECT_ROOT"
echo "SCRIPT_PATH  : $SCRIPT_PATH"
echo "OUT_ROOT     : $OUT_ROOT"
echo "PREFIX       : $PREFIX"
echo "SIZE         : ${WIDTH}x${HEIGHT}"
echo "FPS          : $FPS"
echo "DISPLAY      : ${DISPLAY:-none}"
echo "============================================================"

cd "$PROJECT_ROOT"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] script not found: $SCRIPT_PATH"
    echo "저장소 파일이 누락되었는지 확인하세요."
    exit 1
fi

mkdir -p "$OUT_ROOT"

# pyrealsense2 / cv2 import 확인
python3 - << 'PY'
import sys
try:
    import cv2
    import pyrealsense2 as rs
except Exception as e:
    print("[ERROR] Python import failed:", e)
    print("현재 python3 환경에 opencv-python 또는 pyrealsense2가 없을 수 있습니다.")
    sys.exit(1)
print("[OK] cv2 / pyrealsense2 import success")
PY

python3 "$SCRIPT_PATH" \
    --out-root "$OUT_ROOT" \
    --prefix "$PREFIX" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" \
    "$@"
