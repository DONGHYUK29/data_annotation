import json
from pipeline import common as C
from pipeline.step1_bbox import run as step1
from pipeline.step2_crop import run as step2
from pipeline.step3_seg import run as step3

YOLO_MODELS = {
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
}

def summarize(times):
    avg = sum(times) / len(times) if times else 0.0
    return {
        "per_image_time": times,
        "total_time": sum(times),
        "avg_time": avg,
        "fps": (1.0 / avg) if avg > 0 else 0.0
    }

if __name__ == "__main__":
    C.set_output_root("benchmark")

    for model_name, weight in YOLO_MODELS.items():
        print(f"\n=== BENCHMARK START: {model_name} ===")

        bbox_times = step1(weight, save_output=False)
        crop_times = step2(save_output=False)
        seg_times  = step3(save_output=False)

        report = {
            "model": model_name,
            "bbox": summarize(bbox_times),
            "crop": summarize(crop_times),
            "seg": summarize(seg_times),
        }

        report["pipeline"] = {
            "avg_time": (
                report["bbox"]["avg_time"]
              + report["crop"]["avg_time"]
              + report["seg"]["avg_time"]
            )
        }
        report["pipeline"]["fps"] = (
            1.0 / report["pipeline"]["avg_time"]
            if report["pipeline"]["avg_time"] > 0 else 0.0
        )

        out_path = C.OUT_FPS / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print(
            f"[RESULT] {model_name} | "
            f"BBOX FPS={report['bbox']['fps']:.2f}, "
            f"SEG FPS={report['seg']['fps']:.2f}, "
            f"PIPE FPS={report['pipeline']['fps']:.2f}"
        )

    print("\n[OK] Benchmark finished")
