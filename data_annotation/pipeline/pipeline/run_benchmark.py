# pipeline/run_benchmark.py
import json
from pipeline import common as C
from pipeline.step1_bbox import run as step1
from pipeline.step2_crop import run as step2
from pipeline.step3_seg import run as step3

YOLO_MODELS = {
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
}

IMG_SIZES = {
    "640x640": (640, 640),
    "640x480": (640, 480),
    "768x768": (768, 768),
    "832x832": (832, 832),
    "960x720": (960, 720),
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

    print("\n[WARMUP] Running dummy warm-up pipeline...")

    warmup_weight = "yolo11m.pt"
    first_imgsz = next(iter(IMG_SIZES.values()))
    _ = step1(warmup_weight, imgsz=first_imgsz, save_output=False)
    _ = step2(save_output=False)
    _ = step3(save_output=False)

    print("[WARMUP] Done. Benchmark starts now.\n")

    benchmark_results = []
    for size_name, imgsz in IMG_SIZES.items():

        print(f"\n================ IMG SIZE: {size_name} =================")

        _ = step1("yolo11m.pt", imgsz=imgsz, save_output=False)
        _ = step2(save_output=False)
        _ = step3(save_output=False)

        benchmark_results = []

        for model_name, weight in YOLO_MODELS.items():
            print(f"\n--- {model_name} ---")

            bbox_times = step1(weight, imgsz=imgsz, save_output=False)
            crop_times = step2(save_output=False)
            seg_times  = step3(save_output=False)

            report = {
                "model": model_name,
                "imgsz": imgsz,
                "bbox": summarize(bbox_times),
                "crop": summarize(crop_times),
                "seg": summarize(seg_times),
            }

            pipeline_avg = (
                report["bbox"]["avg_time"]
                + report["crop"]["avg_time"]
                + report["seg"]["avg_time"]
            )

            report["pipeline"] = {
                "avg_time": pipeline_avg,
                "fps": 1.0 / pipeline_avg if pipeline_avg > 0 else 0.0
            }

            out_path = C.OUT_FPS / f"{model_name}_{size_name}.json"
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)

            benchmark_results.append({
                "model": model_name,
                "bbox_fps": report["bbox"]["fps"],
                "crop_fps": report["crop"]["fps"],
                "seg_fps": report["seg"]["fps"],
                "pipe_fps": report["pipeline"]["fps"],
            })

        # ================= SUMMARY 출력 =================
        print(f"\n===== SUMMARY @ IMG SIZE {size_name} =====")
        print(f"{'Model':10s} | {'BBOX FPS':8s} | {'CROP FPS':8s} | {'SEG FPS':7s} | {'PIPE FPS':8s}")
        print("-" * 60)

        for r in benchmark_results:
            print(
                f"{r['model']:10s} | "
                f"{r['bbox_fps']:8.2f} | "
                f"{r['crop_fps']:8.2f} | "
                f"{r['seg_fps']:7.2f} | "
                f"{r['pipe_fps']:8.2f}"
            )

        print("=" * 60)
