# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import cv2
import numpy as np
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*aug_scale_pos_bg*")
    parser.add_argument("--max-num", type=int, default=60)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    paths = sorted(img_dir.glob(args.pattern))

    if len(paths) == 0:
        raise RuntimeError(f"No images found: {img_dir}/{args.pattern}")

    random.seed(42)
    random.shuffle(paths)
    paths = paths[:args.max_num]

    thumb_w, thumb_h = 240, 180
    cols = args.cols
    rows = int(np.ceil(len(paths) / cols))

    sheet = np.ones((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8) * 255

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] failed to read: {p}")
            continue

        img = cv2.resize(img, (thumb_w, thumb_h))

        cv2.putText(
            img,
            p.name[:42],
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        r = i // cols
        c = i % cols
        y1 = r * thumb_h
        x1 = c * thumb_w
        sheet[y1:y1 + thumb_h, x1:x1 + thumb_w] = img

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"[SAVED] {out}")
    print(f"[NUM] {len(paths)} images used")


if __name__ == "__main__":
    main()
