# runs/seg.py
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import pipeline.common as C
import re

def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path.stem)]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a YOLO segmentation model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", type=Path, default=C.INPUT_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    C.INPUT_DIR = args.input_dir.expanduser().resolve()
    C.set_output_root("output_seg")

    model = YOLO(args.model)

    img_paths = sorted(C.list_input_images(), key=natural_key)
    img_count = len(img_paths)
    assert img_count > 0, "No input images found"

    print(f"[SEG ONLY] images: {img_count}")

    t0 = time.time()

    for img_path in img_paths:
        image_id = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 🔥 conf 낮춰서 디버깅
        results = model(
            img,
            verbose=False,
            retina_masks=True,
            conf=0.01
        )[0]

        # detection 없으면 skip
        if results.boxes is None or len(results.boxes) == 0:
            print(f"{image_id} → objects: 0")
            continue

        boxes = results.boxes
        masks = results.masks

        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        print(f"{image_id} → objects: {len(classes)}")

        for i in range(len(classes)):
            class_id = classes[i]
            class_name = results.names[class_id]
            conf = confs[i]

            print(f"  obj{i}: class={class_name}, conf={conf:.3f}")

            # mask 처리
            mask = masks.data[i].cpu().numpy()
            mask = (mask > 0.5).astype("uint8") * 255
            mask = cv2.resize(
                mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # 🔥 class 이름 포함해서 저장
            mask_name = f"{image_id}_obj{i}_{class_name}_{conf:.2f}.png"
            cv2.imwrite(str(C.OUT_IMG / "04_seg" / mask_name), mask)

            # overlay
            overlay = img.copy().astype(np.float32)
            overlay[mask > 0] = (
                0.5 * overlay[mask > 0]
                + 0.5 * np.array([0, 255, 0], dtype=np.float32)
            )
            overlay = overlay.astype(np.uint8)

            overlay_name = f"{image_id}_overlay_{i}_{class_name}_{conf:.2f}.jpg"
            cv2.imwrite(
                str(C.OUT_IMG / "04_seg" / overlay_name),
                overlay
            )

    t1 = time.time()
    total_time = t1 - t0
    fps = img_count / total_time

    print(f"[SEG ONLY] total time: {total_time:.3f}s")
    print(f"[SEG ONLY] FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
