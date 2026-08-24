import argparse
import time
import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Headless RealSense YOLO RGB-D inference")
parser.add_argument("--model", required=True, type=Path, help="YOLO segmentation .pt file")
args = parser.parse_args()

model = YOLO(str(args.model.expanduser().resolve()))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

frame_count = 0
start_time = time.time()
fps = 0.0

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        results = model(frame, verbose=False)

        objects = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = model.names[cls]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                depth = depth_frame.get_distance(cx, cy)
                if depth == 0:
                    continue

                X = (cx - intr.ppx) / intr.fx * depth
                Y = (cy - intr.ppy) / intr.fy * depth
                Z = depth

                objects.append({
                    "class": cls_name,
                    "conf": conf,
                    "center": [cx, cy],
                    "xyz": [float(X), float(Y), float(Z)]
                })

        frame_count += 1
        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

            payload = {
                "fps": float(fps),
                "objects": objects
            }

            print(json.dumps(payload, ensure_ascii=False), flush=True)

except KeyboardInterrupt:
    print("Stopped")

finally:
    pipeline.stop()
