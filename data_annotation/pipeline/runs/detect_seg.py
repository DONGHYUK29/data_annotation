# runs/detect_seg.py
import argparse
import time
from pathlib import Path
import pipeline.common as C
from pipeline.step1_bbox import run as run_bbox
from pipeline.step2_crop import run as run_crop
from pipeline.step3_seg import run as run_seg


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detection followed by SAM segmentation")
    parser.add_argument("--yolo-model", required=True, type=Path)
    parser.add_argument("--sam-model", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, default=C.INPUT_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    C.INPUT_DIR = args.input_dir.expanduser().resolve()
    C.set_output_root("output_detect_seg")

    img_count = len(C.list_input_images())
    assert img_count > 0, "No input images found"

    print(f"[DETECT+SEG] images: {img_count}")

    t0 = time.time()
    run_bbox(model_name=str(args.yolo_model.expanduser().resolve()))
    run_crop()
    run_seg(sam_checkpoint=args.sam_model)
    t1 = time.time()

    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[DETECT+SEG] total time: {total_time:.3f}s")
    print(f"[DETECT+SEG] FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
