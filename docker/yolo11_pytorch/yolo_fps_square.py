import os
import time
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

# =========================
# 설정
# =========================
WEIGHTS_DIR = Path(os.environ.get("YOLO_WEIGHTS_DIR", ".")).expanduser().resolve()
MODELS = OrderedDict({
    "YOLO11s": str(WEIGHTS_DIR / "yolo11s.pt"),
    "YOLO11m": str(WEIGHTS_DIR / "yolo11m.pt"),
    "YOLO11l": str(WEIGHTS_DIR / "yolo11l.pt"),
})

INPUT_SIZES = [320, 480, 640, 800]
BASE_SIZE = 640  # 단락 1에서 사용할 기준 해상도

WARMUP_ITERS = 30
MEASURE_ITERS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark(model, size):
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

    for _ in range(WARMUP_ITERS):
        _ = model(img, imgsz=size, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(MEASURE_ITERS):
        _ = model(img, imgsz=size, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    return MEASURE_ITERS / (t1 - t0)


def main():
    print("=" * 70)
    print("YOLO FPS Benchmark (Square Input)")
    print(f"Device: {DEVICE}")
    print(f"Warmup: {WARMUP_ITERS}, Measure: {MEASURE_ITERS}")
    print("=" * 70)

    results = {name: {} for name in MODELS}

    # =========================
    # Benchmark
    # =========================
    for name, weight in MODELS.items():
        print(f"\n[LOAD] {name} ({weight})")
        model = YOLO(weight).to(DEVICE)

        for size in INPUT_SIZES:
            fps = benchmark(model, size)
            results[name][size] = fps
            print(f"  {name:7s} @ {size}x{size}: {fps:.2f} FPS")

    # =========================
    # 단락 1: 모델 간 비교 (640x640)
    # =========================
    print("\n" + "=" * 70)
    print(f"[1] Model-wise FPS Comparison @ {BASE_SIZE}x{BASE_SIZE}")
    print("-" * 70)
    for name in MODELS:
        print(f"{name:7s}: {results[name][BASE_SIZE]:.2f} FPS")

    # =========================
    # 단락 2: 모델별 사이즈 변화
    # =========================
    print("\n" + "=" * 70)
    print("[2] Resolution-wise FPS per Model")
    print("-" * 70)

    for name in MODELS:
        print(f"\n{name}")
        for size in INPUT_SIZES:
            print(f"  {size:4d}x{size:4d}: {results[name][size]:.2f} FPS")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
