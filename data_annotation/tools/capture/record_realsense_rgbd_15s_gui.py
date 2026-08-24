# -*- coding: utf-8 -*-
"""
record_realsense_rgbd_15s_gui.py

RealSense RGB-D 15초 수집용 GUI 스크립트.

기능:
- RealSense RGB preview
- Depth를 RGB frame에 align
- START를 누르면 최대 15초 동안 0.25초 간격으로 RGB/Depth frame pair 저장
- STOP을 누르면 15초 전이라도 현재 clip 저장 종료
- QUIT 또는 q/ESC로 종료
- s: start
- e: stop
- q/ESC: quit

저장 구조:
OUT_ROOT/session_YYYYMMDD_HHMMSS/
  clip_001_YYYYMMDD_HHMMSS/
    frame_000001_t000000ms_rgb.png
    frame_000001_t000000ms_depth.npy
    frame_000002_t000250ms_rgb.png
    frame_000002_t000250ms_depth.npy
    metadata.json

Depth npy는 기본적으로 RealSense raw depth unit(uint16)을 저장한다.
metadata.json의 depth_scale_m_per_unit 값을 곱하면 meter 단위 depth가 된다.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


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
    color = (0, 0, 255) if active else (255, 255, 255)
    fill = (60, 60, 180) if active else (45, 45, 45)

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


def colorize_depth(depth_raw: np.ndarray, depth_scale: float, max_depth_m: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale
    normalized = np.clip(depth_m / max(max_depth_m, 1e-6), 0.0, 1.0)
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def make_preview(color: np.ndarray, depth_vis: np.ndarray) -> np.ndarray:
    if depth_vis.shape[:2] != color.shape[:2]:
        depth_vis = cv2.resize(depth_vis, (color.shape[1], color.shape[0]))
    return np.hstack([color, depth_vis])


def draw_ui(
    preview: np.ndarray,
    recording: bool,
    total_frame_count: int,
    saved_pair_count: int,
    clip_dir: Optional[Path],
    avg_fps: float,
    remaining_sec: float,
    max_duration_sec: float,
) -> np.ndarray:
    vis = preview.copy()

    draw_button(vis, BTN_START, "START", active=recording)
    draw_button(vis, BTN_STOP, "STOP", active=False)
    draw_button(vis, BTN_QUIT, "QUIT", active=False)

    status = "REC" if recording else "PREVIEW"
    status_color = (0, 0, 255) if recording else (0, 255, 255)
    if recording:
        time_text = f"remaining={max(0.0, remaining_sec):.1f}s/{max_duration_sec:.1f}s"
    else:
        time_text = f"max_duration={max_duration_sec:.1f}s"

    cv2.putText(
        vis,
        (
            f"{status} | total_frames={total_frame_count} | "
            f"saved_pairs={saved_pair_count} | fps={avg_fps:.1f} | {time_text}"
        ),
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    if clip_dir is not None:
        cv2.putText(
            vis,
            f"saving: {clip_dir.name}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        "left: RGB | right: aligned depth    keyboard: s=start, e=stop, q/ESC=quit",
        (20, vis.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


class RgbdFrameRecorder:
    def __init__(
        self,
        out_root: Path,
        width: int,
        height: int,
        fps: int,
        prefix: str,
        max_duration_sec: float,
        save_interval_sec: float,
        depth_scale: float,
        save_depth_meters: bool,
    ) -> None:
        self.out_root = out_root
        self.width = width
        self.height = height
        self.fps = fps
        self.prefix = prefix
        self.max_duration_sec = max_duration_sec
        self.save_interval_sec = save_interval_sec
        self.depth_scale = depth_scale
        self.save_depth_meters = save_depth_meters

        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = out_root / f"session_{session_stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.recording = False
        self.clip_idx = 0
        self.clip_frame_count = 0
        self.clip_seen_frame_count = 0
        self.clip_start_time = 0.0
        self.next_save_elapsed_sec = 0.0
        self.clip_dir: Optional[Path] = None

    def start(self) -> None:
        if self.recording:
            print("[WARN] already recording")
            return

        self.clip_idx += 1
        self.clip_frame_count = 0
        self.clip_seen_frame_count = 0
        self.clip_start_time = time.time()
        self.next_save_elapsed_sec = 0.0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.clip_dir = self.session_dir / f"{self.prefix}_{self.clip_idx:03d}_{stamp}"
        self.clip_dir.mkdir(parents=True, exist_ok=True)

        self.recording = True
        print("=" * 70)
        print("[START RGB-D CAPTURE]")
        print(f"clip dir       : {self.clip_dir}")
        print(f"max duration   : {self.max_duration_sec:.1f} sec")
        print(f"save interval  : {self.save_interval_sec:.3f} sec")
        print(f"rgb format     : png")
        print(f"depth format   : npy ({'meter float32' if self.save_depth_meters else 'raw uint16'})")
        print("=" * 70)

    def elapsed(self) -> float:
        if not self.recording:
            return 0.0
        return time.time() - self.clip_start_time

    def remaining(self) -> float:
        if not self.recording:
            return self.max_duration_sec
        return self.max_duration_sec - self.elapsed()

    def should_auto_stop(self) -> bool:
        return self.recording and self.elapsed() >= self.max_duration_sec

    def write(self, color_bgr: np.ndarray, depth_raw: np.ndarray) -> None:
        if not self.recording:
            return
        if self.clip_dir is None:
            return

        self.clip_seen_frame_count += 1
        elapsed = self.elapsed()
        if elapsed + 1e-6 < self.next_save_elapsed_sec:
            return

        self.clip_frame_count += 1
        elapsed_ms = int(round(elapsed * 1000.0))
        stem = f"frame_{self.clip_frame_count:06d}_t{elapsed_ms:06d}ms"
        rgb_path = self.clip_dir / f"{stem}_rgb.png"
        depth_path = self.clip_dir / f"{stem}_depth.npy"

        ok = cv2.imwrite(str(rgb_path), color_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write RGB image: {rgb_path}")

        if self.save_depth_meters:
            depth_to_save = depth_raw.astype(np.float32) * self.depth_scale
        else:
            depth_to_save = depth_raw.astype(np.uint16, copy=False)
        np.save(depth_path, depth_to_save)

        self.next_save_elapsed_sec += self.save_interval_sec
        if self.next_save_elapsed_sec < elapsed:
            self.next_save_elapsed_sec = elapsed + self.save_interval_sec

    def stop(self, reason: str = "manual") -> None:
        if not self.recording:
            print("[WARN] not recording")
            return

        elapsed = self.elapsed()
        metadata = {
            "clip_dir": str(self.clip_dir),
            "width": self.width,
            "height": self.height,
            "fps_requested": self.fps,
            "max_duration_sec": self.max_duration_sec,
            "save_interval_sec": self.save_interval_sec,
            "elapsed_sec": elapsed,
            "camera_frames_seen": self.clip_seen_frame_count,
            "saved_pairs": self.clip_frame_count,
            "avg_saved_fps": self.clip_frame_count / elapsed if elapsed > 0 else 0.0,
            "rgb_format": "png",
            "depth_format": "npy",
            "depth_saved_as": "meter_float32" if self.save_depth_meters else "raw_uint16",
            "depth_scale_m_per_unit": self.depth_scale,
            "stop_reason": reason,
        }
        if self.clip_dir is not None:
            metadata_path = self.clip_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print("=" * 70)
        print("[STOP RGB-D CAPTURE]")
        print(f"reason     : {reason}")
        print(f"saved dir  : {self.clip_dir}")
        print(f"seen frames: {self.clip_seen_frame_count}")
        print(f"saved pairs: {self.clip_frame_count}")
        print(f"elapsed sec: {elapsed:.1f}")
        if elapsed > 0:
            print(f"avg fps    : {self.clip_frame_count / elapsed:.1f}")
        print("=" * 70)

        self.recording = False
        self.clip_frame_count = 0
        self.clip_seen_frame_count = 0

    def close(self) -> None:
        if self.recording:
            self.stop(reason="close")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "rgbd_15s_captures"),
        help="RGB-D frame pair 저장 root 디렉토리",
    )
    parser.add_argument("--prefix", type=str, default="rgbd")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-sec", type=float, default=15.0)
    parser.add_argument(
        "--save-interval-sec",
        type=float,
        default=0.25,
        help="RGB-D pair 저장 간격. 기본 0.25초는 15초 1회전 기준 약 6도 간격.",
    )
    parser.add_argument(
        "--depth-preview-max-m",
        type=float,
        default=2.0,
        help="preview color map에서 최대 depth로 볼 거리(m)",
    )
    parser.add_argument(
        "--save-depth-meters",
        action="store_true",
        help="지정하면 depth npy를 meter 단위 float32로 저장한다. 기본은 raw uint16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    state = {"action": None}

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if inside_button(x, y, BTN_START):
            state["action"] = "start"
        elif inside_button(x, y, BTN_STOP):
            state["action"] = "stop"
        elif inside_button(x, y, BTN_QUIT):
            state["action"] = "quit"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    window_name = "RealSense RGB-D 15s Recorder"
    recorder: Optional[RgbdFrameRecorder] = None

    print("=" * 70)
    print("RealSense RGB-D 15s Recorder")
    print("=" * 70)
    print(f"out_root    : {out_root}")
    print(f"size        : {args.width}x{args.height}")
    print(f"fps         : {args.fps}")
    print(f"duration    : {args.duration_sec:.1f} sec")
    print(f"interval    : {args.save_interval_sec:.3f} sec")
    print("START: click START or press s")
    print("STOP : click STOP or press e")
    print("QUIT : click QUIT or press q/ESC")
    print("=" * 70)

    total_frame_count = 0
    start_time = time.time()
    pipeline_started = False

    try:
        profile = pipeline.start(config)
        pipeline_started = True
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        recorder = RgbdFrameRecorder(
            out_root=out_root,
            width=args.width,
            height=args.height,
            fps=args.fps,
            prefix=args.prefix,
            max_duration_sec=args.duration_sec,
            save_interval_sec=max(args.save_interval_sec, 1e-3),
            depth_scale=depth_scale,
            save_depth_meters=args.save_depth_meters,
        )

        print(f"depth_scale : {depth_scale}")
        print(f"session_dir : {recorder.session_dir}")
        print("=" * 70)

        for _ in range(30):
            pipeline.wait_for_frames()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            total_frame_count += 1
            elapsed_total = time.time() - start_time
            avg_fps = total_frame_count / max(elapsed_total, 1e-6)

            action = state.get("action")
            state["action"] = None
            if action == "start":
                recorder.start()
            elif action == "stop":
                recorder.stop(reason="manual")
            elif action == "quit":
                break

            if recorder.recording:
                recorder.write(color_bgr, depth_raw)
                if recorder.should_auto_stop():
                    recorder.stop(reason="duration_complete")

            depth_vis = colorize_depth(depth_raw, depth_scale, args.depth_preview_max_m)
            preview = make_preview(color_bgr, depth_vis)
            vis = draw_ui(
                preview=preview,
                recording=recorder.recording,
                total_frame_count=total_frame_count,
                saved_pair_count=recorder.clip_frame_count,
                clip_dir=recorder.clip_dir,
                avg_fps=avg_fps,
                remaining_sec=recorder.remaining(),
                max_duration_sec=args.duration_sec,
            )
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                recorder.start()
            elif key == ord("e"):
                recorder.stop(reason="manual")
            elif key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by Ctrl+C")

    finally:
        if recorder is not None:
            recorder.close()
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()

        print("=" * 70)
        print("CLOSED")
        if recorder is not None:
            print(f"session_dir: {recorder.session_dir}")
            print("saved clips:")
            for p in sorted(recorder.session_dir.glob("*")):
                if p.is_dir():
                    rgb_count = len(list(p.glob("*_rgb.png")))
                    depth_count = len(list(p.glob("*_depth.npy")))
                    print(f"  - {p} | rgb={rgb_count}, depth={depth_count}")
        print("=" * 70)


if __name__ == "__main__":
    main()
