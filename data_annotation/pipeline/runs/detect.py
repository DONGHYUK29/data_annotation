# runs/detect.py
import argparse
import time
from pathlib import Path
import pipeline.common as C
from pipeline.step1_bbox import run as run_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO bbox detection")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, default=C.INPUT_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    C.INPUT_DIR = args.input_dir.expanduser().resolve()
    C.set_output_root("output_detect")

    img_count = len(C.list_input_images())
    assert img_count > 0, "No input images found"

    print(f"[DETECT] images: {img_count}")

    t0 = time.time()
    run_bbox(model_name=str(args.model.expanduser().resolve()))
    t1 = time.time()

    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[DETECT] total time: {total_time:.3f}s")
    print(f"[DETECT] FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
