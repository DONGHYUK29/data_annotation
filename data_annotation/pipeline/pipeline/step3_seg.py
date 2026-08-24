# pipeline/step3_seg.py
import json
import cv2
import torch
import numpy as np
from pathlib import Path
import pipeline.common as C
from pipeline.common import Timer
from segment_anything import sam_model_registry, SamPredictor


def run(save_output=True, sam_checkpoint=None):
    checkpoint = Path(sam_checkpoint).expanduser().resolve() if sam_checkpoint else (
        C.BASE_DIR / "weights" / "sam_vit_b_01ec64.pth"
    )
    sam = sam_model_registry["vit_b"](
        checkpoint=str(checkpoint)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)

    times = []

    for rec_path in C.CACHE_REC.glob("*.json"):
        with Timer() as t:
            with open(rec_path) as f:
                rec = json.load(f)

            if save_output:
                seg_txt = C.OUT_META / "seg" / f"{Path(rec_path).stem}.txt"
                sf = open(seg_txt, "w")
                sf.write("instance_id,mask_image\n")

            for inst in rec["instances"]:

                # 🔥 crop 있으면 crop, 없으면 원본
                img_path = inst.get("crop_path", rec["image_path"])
                img = cv2.imread(img_path)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                predictor.set_image(rgb)

                # 🔥 box 선택
                if "crop_box" in inst:
                    box = np.array(inst["crop_box"])
                else:
                    x1, y1, x2, y2 = inst["bbox"]
                    box = np.array([x1, y1, x2, y2])

                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False
                )

                mask = (masks[0] * 255).astype("uint8")

                name = f"{inst['instance_id']}.png"

                # mask 저장
                cv2.imwrite(str(C.CACHE_MASK / name), mask)

                if save_output:
                    overlay = img.copy().astype(np.float32)
                    overlay[mask > 0] = (
                        0.5 * overlay[mask > 0]
                        + 0.5 * np.array([0, 255, 0], dtype=np.float32)
                    )
                    overlay = overlay.astype(np.uint8)

                    cv2.imwrite(str(C.OUT_IMG / "04_seg" / name), overlay)
                    sf.write(f"{inst['instance_id']},{name}\n")

            if save_output:
                sf.close()

        times.append(t.elapsed)

    return times
