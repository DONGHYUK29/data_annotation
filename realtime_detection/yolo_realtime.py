import argparse
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import socket
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="RealSense YOLO RGB-D inference with TCP JSON output")
parser.add_argument("--model", required=True, type=Path, help="YOLO segmentation .pt file")
parser.add_argument("--socket-host", default="127.0.0.1")
parser.add_argument("--socket-port", default=5000, type=int)
args = parser.parse_args()

# ==============================
# 🔥 모델 경로
# ==============================
model = YOLO(str(args.model.expanduser().resolve()))

# ==============================
# 🔥 SOCKET 설정
# ==============================
WSL_IP = args.socket_host
PORT = args.socket_port

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((WSL_IP, PORT))

# ==============================
# 🔥 RealSense 설정 (RGB + Depth)
# ==============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# intrinsic 가져오기
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# ==============================
# 🔥 화면 설정
# ==============================
cv2.namedWindow("RealSense YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RealSense YOLO", 1280, 960)

# ==============================
# 🔥 FPS 변수
# ==============================
frame_count = 0
start_time = time.time()
fps = 0

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # ==============================
        # YOLO 추론
        # ==============================
        results = model(frame, verbose=False)

        annotated = frame.copy()
        objects = []

        for r in results:

            # ===== segmentation =====
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                for mask in masks:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    green = np.zeros_like(annotated)
                    green[:, :] = (0, 255, 0)

                    alpha = 0.4
                    annotated[mask > 0.5] = (
                        annotated[mask > 0.5] * (1 - alpha) + green[mask > 0.5] * alpha
                    )

            # ===== bbox =====
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    cls_name = model.names[cls]

                    # 중심점
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # depth
                    depth = depth_frame.get_distance(cx, cy)

                    if depth == 0:
                        continue

                    # 3D 좌표 변환
                    X = (cx - intr.ppx) / intr.fx * depth
                    Y = (cy - intr.ppy) / intr.fy * depth
                    Z = depth

                    # 시각화
                    label = f"{cls_name} {conf:.2f} Z:{Z:.2f}m"

                    margin = 0

                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(annotated, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    cv2.circle(annotated, (cx, cy), 3, (0,0,255), -1)

                    # JSON용 데이터
                    objects.append({
                        "class": cls_name,
                        "conf": float(conf),
                        "xyz": [float(X), float(Y), float(Z)]
                    })

        # ==============================
        # FPS 계산
        # ==============================
        frame_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # ==============================
        # socket 전송
        # ==============================
        data = {
            "fps": float(fps),
            "objects": objects
        }

        try:
            sock.sendall((json.dumps(data) + "\n").encode("utf-8"))
        except:
            pass  # 연결 끊겨도 죽지 않게

        # ==============================
        # FPS 표시
        # ==============================
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # ==============================
        # 출력
        # ==============================
        cv2.imshow("RealSense YOLO", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    sock.close()
    cv2.destroyAllWindows()
