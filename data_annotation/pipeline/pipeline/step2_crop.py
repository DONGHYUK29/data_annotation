# pipeline/step2_crop.py
import json
import cv2
import pipeline.common as C
from pipeline.common import Timer


def run(save_output=True):
    times = []

    for rec_path in C.CACHE_REC.glob("*.json"):
        with Timer() as t:
            with open(rec_path) as f:
                rec = json.load(f)

            img = cv2.imread(rec["image_path"])
            if img is None:
                continue

            for inst in rec["instances"]:
                x1, y1, x2, y2 = inst["bbox"]

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                name = f"{inst['instance_id']}.png"
                crop_path = C.CACHE_CROP / name

                # crop 저장
                cv2.imwrite(str(crop_path), crop)

                # crop 관련 정보 저장
                inst["crop_path"] = str(crop_path)
                inst["crop_box"] = [0, 0, x2 - x1, y2 - y1]  # 🔥 SAM용 bbox (crop 기준)

                if save_output:
                    cv2.imwrite(str(C.OUT_IMG / "03_crop" / name), crop)

            # rec 업데이트
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=2)

        times.append(t.elapsed)

    return times
