import os
import time
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

# =========================
# 설정 (현장 튜닝용)
# =========================

WEIGHTS_DIR = Path(os.environ.get("YOLO_WEIGHTS_DIR", ".")).expanduser().resolve()
MODELS = OrderedDict({
    "YOLO11s": str(WEIGHTS_DIR / "yolo11s.pt"),
    "YOLO11m": str(WEIGHTS_DIR / "yolo11m.pt"),
    "YOLO11l": str(WEIGHTS_DIR / "yolo11l.pt"),
})

# (width, height) 형태로 명시
INPUT_SIZES = [
    (640, 480),    # 산업용 카메라 표준 (4:3)
    (1280, 720),   # HD (16:9)
    (480, 480),    # ROI crop
    (640, 640),    # YOLO 표준
    (320, 240),    # 초고속 제어용
]

WARMUP_ITERS = 30
MEASURE_ITERS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark(model, w, h):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = model(img, imgsz=(h, w), verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(MEASURE_ITERS):
        _ = model(img, imgsz=(h, w), verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    return MEASURE_ITERS / (t1 - t0)



def main():
    print("=" * 60)
    print("YOLO FPS Benchmark (Rectangular Input)")
    print(f"Device: {DEVICE}")
    print(f"Warmup iters: {WARMUP_ITERS}, Measure iters: {MEASURE_ITERS}")
    print("=" * 60)

    results = {name: {} for name in MODELS}

    for name, weight in MODELS.items():
        print(f"\n[LOAD] {name} ({weight})")
        model = YOLO(weight).to(DEVICE)

        for (w, h) in INPUT_SIZES:
            key = f"{w}x{h}"
            print(
                f"  → Benchmarking {name} @ {w}x{h} ... ",
                end="",
                flush=True,
            )
            fps = benchmark(model, w, h)
            results[name][key] = fps
            print(f"{fps:.2f} FPS")

    # =========================
    # 결과 테이블 출력
    # =========================

    print("\n" + "=" * 60)
    print("Final FPS Table (FPS)")

    header = "model \\ input_size | " + " | ".join(
        f"{w}x{h: <3}" for (w, h) in INPUT_SIZES
    )
    print(header)
    print("-" * len(header))

    for name in MODELS:
        row = f"{name:<16} | " + " | ".join(
            f"{results[name][f'{w}x{h}']:6.2f}"
            for (w, h) in INPUT_SIZES
        )
        print(row)

    print("=" * 60)


if __name__ == "__main__":
    main()
