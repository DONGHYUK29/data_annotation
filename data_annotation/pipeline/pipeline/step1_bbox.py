# pipeline/step1_bbox.py
import cv2, json
from ultralytics import YOLO
import pipeline.common as C
from pipeline.common import Timer

def run(model_name="yolo26l.pt", imgsz=640, save_output=True):
    model = YOLO(str(C.BASE_DIR / "weights" / model_name))
    times = []

    for img_path in C.list_input_images():
        image_id = img_path.stem

        with Timer() as t:
            img = cv2.imread(str(img_path))
            imgsz_arg = list(imgsz) if isinstance(imgsz, tuple) else imgsz
            results = model(img, imgsz=imgsz_arg, verbose=False)[0]

            instances = []
            if results.boxes is not None:
                for i, box in enumerate(results.boxes):
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    instances.append({
                        "instance_id": f"{image_id}_obj{i}",
                        "class_id": cls,
                        "confidence": conf,
                        "bbox": [x1,y1,x2,y2],
                    })

                    if save_output:
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

            if save_output:
                cv2.imwrite(str(C.OUT_IMG / "01_input" / img_path.name), img)
                cv2.imwrite(str(C.OUT_IMG / "02_bbox" / f"{image_id}.jpg"), img)

                with open(C.OUT_META / "bbox" / f"{image_id}.txt", "w") as f:
                    f.write("instance_id,class_id,x1,y1,x2,y2,confidence\n")
                    for inst in instances:
                        x1,y1,x2,y2 = inst["bbox"]
                        f.write(
                            f"{inst['instance_id']},{inst['class_id']},"
                            f"{x1},{y1},{x2},{y2},{inst['confidence']}\n"
                        )

            with open(C.CACHE_REC / f"{image_id}.json", "w") as f:
                json.dump({
                    "image_path": str(img_path),
                    "instances": instances
                }, f, indent=2)

        times.append(t.elapsed)

    return times
