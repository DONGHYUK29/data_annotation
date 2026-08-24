# -*- coding: utf-8 -*-
"""
record_realsense_gui.py

RealSense RGB 배경 영상 녹화용 GUI 스크립트.

기능:
- RealSense RGB preview
- OpenCV 창 안의 START / STOP / QUIT 버튼 클릭
- START를 누른 순간부터 녹화
- STOP을 누르면 현재 mp4 저장 완료
- 다시 START를 누르면 새 mp4로 저장
- q 또는 ESC로도 종료 가능
- s: start
- e: stop
- q/ESC: quit

저장 구조:
OUT_ROOT/session_YYYYMMDD_HHMMSS/
  lab_bg_YYYYMMDD_HHMMSS_001.mp4
  lab_bg_YYYYMMDD_HHMMSS_002.mp4
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# Button UI
# ============================================================

BTN_START = (20, 20, 170, 75)
BTN_STOP = (190, 20, 340, 75)
BTN_QUIT = (360, 20, 510, 75)


def inside_button(x: int, y: int, box: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    text: str,
    active: bool = False,
) -> None:
    x1, y1, x2, y2 = box

    if active:
        color = (0, 0, 255)
        fill = (60, 60, 180)
    else:
        color = (255, 255, 255)
        fill = (45, 45, 45)

    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        img,
        text,
        (x1 + 18, y1 + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_ui(
    frame: np.ndarray,
    recording: bool,
    frame_count: int,
    clip_frame_count: int,
    current_path: Optional[Path],
    avg_fps: float,
) -> np.ndarray:
    vis = frame.copy()

    draw_button(vis, BTN_START, "START", active=recording)
    draw_button(vis, BTN_STOP, "STOP", active=False)
    draw_button(vis, BTN_QUIT, "QUIT", active=False)

    status = "REC" if recording else "PREVIEW"
    status_color = (0, 0, 255) if recording else (0, 255, 255)

    cv2.putText(
        vis,
        f"{status} | total_frames={frame_count} | clip_frames={clip_frame_count} | fps={avg_fps:.1f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    if current_path is not None:
        cv2.putText(
            vis,
            f"saving: {current_path.name}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        "keyboard: s=start, e=stop, q/ESC=quit",
        (20, vis.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# Recorder
# ============================================================

class Recorder:
    def __init__(
        self,
        out_root: Path,
        width: int,
        height: int,
        fps: int,
        prefix: str,
    ) -> None:
        self.out_root = out_root
        self.width = width
        self.height = height
        self.fps = fps
        self.prefix = prefix

        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = out_root / f"session_{session_stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.writer: Optional[cv2.VideoWriter] = None
        self.current_path: Optional[Path] = None
        self.recording = False
        self.clip_idx = 0
        self.clip_frame_count = 0
        self.clip_start_time = 0.0

    def start(self) -> None:
        if self.recording:
            print("[WARN] already recording")
            return

        self.clip_idx += 1
        self.clip_frame_count = 0
        self.clip_start_time = time.time()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_path = self.session_dir / f"{self.prefix}_{stamp}_{self.clip_idx:03d}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.current_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self.writer.isOpened():
            self.writer = None
            self.current_path = None
            raise RuntimeError("VideoWriter open failed")

        self.recording = True
        print("=" * 70)
        print("[START REC]")
        print(f"save path: {self.current_path}")
        print("=" * 70)

    def write(self, frame: np.ndarray) -> None:
        if not self.recording:
            return
        if self.writer is None:
            return

        self.writer.write(frame)
        self.clip_frame_count += 1

    def stop(self) -> None:
        if not self.recording:
            print("[WARN] not recording")
            return

        elapsed = time.time() - self.clip_start_time

        if self.writer is not None:
            self.writer.release()

        print("=" * 70)
        print("[STOP REC]")
        print(f"saved      : {self.current_path}")
        print(f"frames     : {self.clip_frame_count}")
        print(f"elapsed sec: {elapsed:.1f}")
        if elapsed > 0:
            print(f"avg fps    : {self.clip_frame_count / elapsed:.1f}")
        print("=" * 70)

        self.writer = None
        self.recording = False
        self.clip_frame_count = 0

    def close(self) -> None:
        if self.recording:
            self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "background_videos"),
        help="녹화 영상 저장 root 디렉토리",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="lab_bg",
        help="저장 영상 파일 prefix",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    state = {
        "action": None,
    }

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if inside_button(x, y, BTN_START):
            state["action"] = "start"
        elif inside_button(x, y, BTN_STOP):
            state["action"] = "stop"
        elif inside_button(x, y, BTN_QUIT):
            state["action"] = "quit"

    recorder = Recorder(
        out_root=out_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        prefix=args.prefix,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    window_name = "RealSense Background Recorder"

    print("=" * 70)
    print("RealSense Background Recorder")
    print("=" * 70)
    print(f"out_root   : {out_root}")
    print(f"session_dir: {recorder.session_dir}")
    print(f"size       : {args.width}x{args.height}")
    print(f"fps        : {args.fps}")
    print("START: click START or press s")
    print("STOP : click STOP or press e")
    print("QUIT : click QUIT or press q/ESC")
    print("=" * 70)

    frame_count = 0
    start_time = time.time()

    try:
        pipeline.start(config)

        # auto exposure 안정화용. 초반 프레임 버림.
        for _ in range(30):
            pipeline.wait_for_frames()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            frame_count += 1
            elapsed_total = time.time() - start_time
            avg_fps = frame_count / max(elapsed_total, 1e-6)

            # action 처리
            action = state.get("action")
            state["action"] = None

            if action == "start":
                recorder.start()
            elif action == "stop":
                recorder.stop()
            elif action == "quit":
                break

            # 녹화 중이면 UI 없는 원본 frame 저장
            recorder.write(frame)

            # 화면에는 UI overlay 표시
            vis = draw_ui(
                frame=frame,
                recording=recorder.recording,
                frame_count=frame_count,
                clip_frame_count=recorder.clip_frame_count,
                current_path=recorder.current_path,
                avg_fps=avg_fps,
            )

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                recorder.start()
            elif key == ord("e"):
                recorder.stop()
            elif key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by Ctrl+C")

    finally:
        recorder.close()
        pipeline.stop()
        cv2.destroyAllWindows()

        print("=" * 70)
        print("CLOSED")
        print(f"session_dir: {recorder.session_dir}")
        print("saved videos:")
        for p in sorted(recorder.session_dir.glob("*.mp4")):
            print(f"  - {p}")
        print("=" * 70)


if __name__ == "__main__":
    main()
