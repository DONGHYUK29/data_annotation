from __future__ import annotations

import io
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
import mimetypes
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image
import time 
from contextlib import asynccontextmanager

import config as cfg
from .sam_backend import predict_sam_mask

cfg.ensure_stage_dirs()

IMAGE_DIR = cfg.IMAGES_RAW_DIR
MASK_DIR = cfg.OUTPUT1_MASKS
LABEL_DIR = cfg.OUTPUT1_LABELS

AFTER_IMAGE_DIR = cfg.OUTPUT2_IMAGES
AFTER_MASK_DIR = cfg.OUTPUT2_MASKS
AFTER_LABEL_DIR = cfg.OUTPUT2_LABELS

@asynccontextmanager
async def lifespan(app: FastAPI):
    def print_notice():
        time.sleep(1)
        print("\n" + "="*65)
        print(f"웹 브라우저를 열고 아래 주소로 접속해 주세요:")
        print(f"👉 http://localhost:{cfg.WEB_PORT}")
        print("="*65 + "\n")
        
    threading.Thread(target=print_notice, daemon=True).start()
    yield

app = FastAPI(title="Annotation Web Edit + SAM Point Assist", lifespan=lifespan)

MASK_ALPHA = 110
MASK_RGBA = (0, 255, 0, MASK_ALPHA)

PIPELINE_STEPS = ("input", "segment", "gui", "export", "train", "clean", "fix-names")

PIPELINE_DIRS = {
    "input_images": cfg.INPUT_IMAGES_DIR,
    "segment_images": cfg.OUTPUT1_DIR / "images",
    "segment_labels": cfg.OUTPUT1_LABELS,
    "segment_masks": cfg.OUTPUT1_MASKS,
    "export_images": cfg.DATASET_DIR / "images",
    "export_labels": cfg.DATASET_DIR / "labels",
    "export_masks": cfg.DATASET_DIR / "masks",
}

JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
JOB_PROCS: dict[str, subprocess.Popen] = {}


def read_img_safe(path: str | Path, flags=cv2.IMREAD_COLOR):
    try:
        with open(str(path), "rb") as f:
            bytes_arr = bytearray(f.read())
        np_arr = np.asarray(bytes_arr, dtype=np.uint8)
        return cv2.imdecode(np_arr, flags)
    except Exception:
        return None


def write_img_safe(path: str | Path, img, params=None):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ext = path.suffix or ".png"
        ok, encoded = cv2.imencode(ext, img, params if params is not None else [])
        if not ok:
            return False

        with open(str(path), "wb") as f:
            f.write(encoded.tobytes())
        return True
    except Exception:
        return False


def ensure_output_dirs() -> None:
    for p in (AFTER_IMAGE_DIR, AFTER_MASK_DIR, AFTER_LABEL_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _run_job(job_id: str, cmd: list[str]) -> None:
    with JOBS_LOCK:
        if JOBS[job_id]["status"] == "stopped":
            return
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["output"] = ""

    try:
        proc = subprocess.Popen(
            " ".join(cmd),
            cwd=str(cfg.PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        with JOBS_LOCK:
            JOB_PROCS[job_id] = proc

        full_output = []

        while True:
            line = proc.stdout.readline()

            if not line and proc.poll() is not None:
                break

            if line:
                full_output.append(line)

                with JOBS_LOCK:
                    JOBS[job_id]["output"] = "".join(full_output)

        return_code = proc.wait()

        with JOBS_LOCK:
            was_stopping = JOBS[job_id]["status"] in ("stopping", "stopped")
            JOBS[job_id]["status"] = "stopped" if was_stopping else (
                "done" if return_code == 0 else "failed"
            )
            JOBS[job_id]["returncode"] = return_code
            JOBS[job_id]["output"] = "".join(full_output)
            JOB_PROCS.pop(job_id, None)

    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["returncode"] = -1
            JOBS[job_id]["output"] = str(exc)
            JOB_PROCS.pop(job_id, None)


def _stop_job_process(job_id: str) -> bool:
    with JOBS_LOCK:
        proc = JOB_PROCS.get(job_id)
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in ("queued", "running", "stopping"):
            return False
        job["status"] = "stopping"

    if not proc or proc.poll() is not None:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "stopped"
        return True

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    return True


def _build_docker_command(step: str, payload: dict) -> list[str]:
    base = ["docker", "compose", "run", "--rm"]
    if step == "gui":
        base.append("--service-ports")
    cmd_step = "edit" if step == "gui" else step
    cmd = ["python", "run.py", cmd_step]

    if step == "segment":
        if payload.get("input_dir"):
            cmd.extend(["--input", str(payload["input_dir"])])
        if payload.get("output_dir"):
            cmd.extend(["--output", str(payload["output_dir"])])
        if payload.get("weights"):
            cmd.extend(["--weights", str(payload["weights"])])
    elif step == "gui":
        cmd.extend(["--host", "0.0.0.0", "--port", str(cfg.WEB_PORT)])
    elif step == "export":
        bg_prefix = str(payload.get("bg_prefix", "")).strip()
        if bg_prefix:
            cmd.extend(["--bg", bg_prefix])
    elif step == "train":
        cmd.extend(["--weights", str(payload["weights"])])
        cmd.extend(["--name", str(payload.get("model_name", "exp"))])
        cmd.extend(["--epochs", str(int(payload.get("epochs", 100)))])
        cmd.extend(["--batch", str(int(payload.get("batch", 16)))])
        cmd.extend(["--imgsz", str(int(payload.get("imgsz", 640)))])
        cmd.extend(["--num-classes", str(int(payload.get("num_classes", 1)))])
        cmd.extend(["--val-ratio", str(float(payload.get("val_ratio", 0.2)))])
        cmd.extend(
            [
                "--data",
                str(payload.get("data") or (cfg.TRAINING_DIR / "dataset.yaml")),
            ]
        )
        cmd.extend(["--project", str(payload.get("project") or cfg.WEIGHTS_DIR)])
        if payload.get("lr0") not in (None, ""):
            cmd.extend(["--lr0", str(float(payload["lr0"]))])
        if payload.get("patience") not in (None, ""):
            cmd.extend(["--patience", str(int(payload["patience"]))])
        if payload.get("optimizer"):
            cmd.extend(["--optimizer", str(payload["optimizer"])])
        if payload.get("freeze") not in (None, ""):
            cmd.extend(["--freeze", str(int(payload["freeze"]))])
        if payload.get("augment"):
            cmd.extend(["--augment", str(payload["augment"])])
        if payload.get("workers") not in (None, ""):
            cmd.extend(["--workers", str(int(payload["workers"]))])
        if payload.get("seed") not in (None, ""):
            cmd.extend(["--seed", str(int(payload["seed"]))])
        if payload.get("resume") not in (None, "", False):
            r = payload["resume"]
            if r is True:
                cmd.append("--resume")
                cmd.append("true")
            else:
                cmd.extend(["--resume", str(r)])
        if payload.get("cache"):
            cmd.extend(["--cache", str(payload["cache"])])
        if payload.get("amp") is True:
            cmd.extend(["--amp", "true"])
        elif payload.get("amp") is False:
            cmd.extend(["--amp", "false"])
        if payload.get("cos_lr") is True:
            cmd.extend(["--cos-lr", "true"])
        elif payload.get("cos_lr") is False:
            cmd.extend(["--cos-lr", "false"])
        if payload.get("weight_decay") not in (None, ""):
            cmd.extend(["--weight-decay", str(float(payload["weight_decay"]))])
        if payload.get("dropout") not in (None, ""):
            cmd.extend(["--dropout", str(float(payload["dropout"]))])
        if payload.get("save_period") not in (None, ""):
            cmd.extend(["--save-period", str(int(payload["save_period"]))])
        if payload.get("pretrained_backbone_only"):
            cmd.append("--pretrained-backbone-only")
    elif step == "clean":
        cmd.extend(["--mode", str(payload.get("clean_mode", "all"))])
    elif step == "fix-names":
        if payload.get("dir"):
            cmd.extend(["--dir", str(payload["dir"])])
            
    return cmd


def natural_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.stem)
    ]


def list_input_images() -> list[Path]:
    files: list[Path] = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
        files.extend(IMAGE_DIR.glob(ext))
    files.sort(key=natural_key)
    return files


def stem_to_class_id(stem: str) -> int:
    parts = stem.split("_")
    try:
        if parts[0].isnumeric():
            return int(parts[0])
        if len(parts) > 1 and parts[1].isnumeric():
            return int(parts[1])
    except Exception:
        pass
    return 0


def read_class_id_from_label(label_path: Path) -> int | None:
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                return int(parts[0])
    except Exception:
        return None
    return None


def get_label_path_to_load(stem: str) -> Path | None:
    edited_label = AFTER_LABEL_DIR / f"{stem}.txt"
    base_label = LABEL_DIR / f"{stem}.txt"

    if edited_label.exists():
        return edited_label
    if base_label.exists():
        return base_label
    return None


def resolve_class_id(stem: str) -> int:
    label_path = get_label_path_to_load(stem)
    if label_path is not None:
        class_id = read_class_id_from_label(label_path)
        if class_id is not None:
            return class_id
    return stem_to_class_id(stem)


def mask_to_bbox_and_polygon(mask_u8: np.ndarray):
    m = (mask_u8 > 127).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (x1, y1, x2, y2), None

    cnt = max(contours, key=cv2.contourArea)
    eps = 0.5
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True).reshape(-1, 2).astype(np.float32)

    if len(approx) < 3:
        poly = cnt.reshape(-1, 2).astype(np.float32)
    else:
        poly = approx

    polygon = [(float(x), float(y)) for x, y in poly]
    return (x1, y1, x2, y2), polygon


def get_image_path_by_stem(stem: str) -> Path | None:
    for ext in ("jpg", "png", "jpeg", "bmp"):
        p = IMAGE_DIR / f"{stem}.{ext}"
        if p.exists():
            return p
    return None


def get_display_image_path(stem: str) -> Path | None:
    edited_img = AFTER_IMAGE_DIR / f"{stem}.png"
    if edited_img.exists():
        return edited_img
    return get_image_path_by_stem(stem)


def get_mask_path_to_load(stem: str) -> Path | None:
    edited_mask = AFTER_MASK_DIR / f"{stem}_edited.png"
    moved_mask = AFTER_MASK_DIR / f"{stem}.png"
    base_mask = MASK_DIR / f"{stem}.png"

    if edited_mask.exists():
        return edited_mask
    if moved_mask.exists():
        return moved_mask
    if base_mask.exists():
        return base_mask
    return None


def has_output1_artifact(stem: str) -> bool:
    return (MASK_DIR / f"{stem}.png").exists() or (LABEL_DIR / f"{stem}.txt").exists()


def count_remaining_files_to_move() -> int:
    pending = 0
    for image_path in list_input_images():
        stem = image_path.stem
        edited_mask = AFTER_MASK_DIR / f"{stem}_edited.png"
        moved_mask = AFTER_MASK_DIR / f"{stem}.png"

        if edited_mask.exists() or moved_mask.exists():
            continue
        if not has_output1_artifact(stem):
            continue

        pending += 1

    return pending


def png_response_from_array(arr: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encoding failed")

    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def move_remaining_files():
    ensure_output_dirs()
    moved = 0

    for image_path in list_input_images():
        stem = image_path.stem
        edited_mask = AFTER_MASK_DIR / f"{stem}_edited.png"
        moved_mask = AFTER_MASK_DIR / f"{stem}.png"

        if edited_mask.exists() or moved_mask.exists():
            continue

        base_mask = MASK_DIR / f"{stem}.png"
        base_label = LABEL_DIR / f"{stem}.txt"

        if not base_mask.exists() and not base_label.exists():
            continue

        dst_image = AFTER_IMAGE_DIR / f"{stem}.png"
        dst_mask = AFTER_MASK_DIR / f"{stem}.png"
        dst_label = AFTER_LABEL_DIR / f"{stem}.txt"

        img = read_img_safe(image_path)
        if img is not None:
            write_img_safe(dst_image, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        if base_mask.exists():
            mask = read_img_safe(base_mask, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.uint8) * 255
                write_img_safe(dst_mask, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        if base_label.exists():
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            with open(base_label, "r", encoding="utf-8") as fsrc, open(dst_label, "w", encoding="utf-8") as fdst:
                fdst.write(fsrc.read())

        moved += 1

    return moved


@app.post("/api/move_remaining")
def api_move_remaining():
    moved = move_remaining_files()
    return {"ok": True, "moved": moved}


@app.get("/api/move_remaining/status")
def api_move_remaining_status():
    return {"ok": True, "pending": count_remaining_files_to_move()}


@app.get("/api/images")
def api_images():
    files = list_input_images()
    return {
        "items": [p.stem for p in files],
        "count": len(files),
    }


@app.get("/api/input/status")
def api_input_status():
    files = list_input_images()
    return {
        "dir": str(IMAGE_DIR),
        "count": len(files),
        "samples": [p.name for p in files[:10]],
    }


@app.post("/api/input/upload")
async def api_input_upload(files: list[UploadFile] = File(...)):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    allowed = {".png", ".jpg", ".jpeg", ".bmp"}
    saved: list[str] = []
    skipped: list[str] = []

    for f in files:
        name = Path(f.filename or "").name
        ext = Path(name).suffix.lower()
        if not name or ext not in allowed:
            skipped.append(name or "(unnamed)")
            continue
        dest = IMAGE_DIR / name
        data = await f.read()
        try:
            dest.write_bytes(data)
            saved.append(name)
        except OSError:
            skipped.append(name)

    return {"ok": True, "saved": saved, "skipped": skipped, "dir": str(IMAGE_DIR)}


@app.get("/api/image/{stem}")
def api_image(stem: str):
    p = get_display_image_path(stem)
    if p is None or not p.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = read_img_safe(p, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image")
    return png_response_from_array(img)


@app.get("/api/mask/{stem}")
def api_mask(stem: str):
    image_path = get_display_image_path(stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")

    img = read_img_safe(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image")
    h, w = img.shape[:2]

    mask_path = get_mask_path_to_load(stem)
    if mask_path is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = read_img_safe(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = (mask > 127).astype(np.uint8) * 255

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = MASK_RGBA[0]
    rgba[..., 1] = MASK_RGBA[1]
    rgba[..., 2] = MASK_RGBA[2]
    rgba[..., 3] = np.where(mask > 0, MASK_RGBA[3], 0).astype(np.uint8)
    return png_response_from_array(rgba)


@app.get("/api/meta/{stem}")
def api_meta(stem: str):
    image_path = get_display_image_path(stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")

    img = read_img_safe(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image")

    h, w = img.shape[:2]
    return {
        "stem": stem,
        "width": w,
        "height": h,
        "class_id": resolve_class_id(stem),
    }


@app.post("/api/sam_predict/{stem}")
async def api_sam_predict(stem: str, request: Request):
    payload = await request.json()
    points = payload.get("points", [])

    if not points:
        raise HTTPException(status_code=400, detail="points are required")

    image_path = get_display_image_path(stem)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    image_bgr = read_img_safe(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read image")

    try:
        point_coords = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)
        point_labels = np.array([int(p["label"]) for p in points], dtype=np.int32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid points payload: {e}")

    try:
        mask = predict_sam_mask(
            image_bgr=image_bgr,
            point_coords=point_coords,
            point_labels=point_labels,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM inference failed: {e}")

    if mask is None:
        raise HTTPException(status_code=500, detail="SAM returned None")

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.ndim != 2:
        raise HTTPException(status_code=500, detail="SAM mask must be HxW")

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return png_response_from_array(mask)


@app.post("/api/save/{stem}")
async def api_save(stem: str, request: Request):
    ensure_output_dirs()

    payload = await request.json()
    mask_data = payload.get("mask")
    if not mask_data:
        raise HTTPException(status_code=400, detail="mask is required")

    image_path = get_image_path_by_stem(stem)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="Original image not found")

    original_img = read_img_safe(image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise HTTPException(status_code=500, detail="Failed to read original image")

    try:
        mask_img = Image.open(io.BytesIO(bytes(mask_data))).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid mask payload: {e}")

    mask = np.array(mask_img)
    mask = (mask > 127).astype(np.uint8) * 255

    h, w = original_img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255

    image_save_path = AFTER_IMAGE_DIR / f"{stem}.png"
    mask_save_path = AFTER_MASK_DIR / f"{stem}_edited.png"
    label_save_path = AFTER_LABEL_DIR / f"{stem}.txt"

    copied_mask_path = AFTER_MASK_DIR / f"{stem}.png"
    if copied_mask_path.exists():
        try:
            copied_mask_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove old copied mask: {e}")

    ok_img = write_img_safe(image_save_path, original_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    ok_mask = write_img_safe(mask_save_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    if not ok_img or not ok_mask:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save image or mask: image={ok_img}, mask={ok_mask}, "
                   f"image_path={image_save_path}, mask_path={mask_save_path}"
        )

    bbox, poly = mask_to_bbox_and_polygon(mask)
    class_id = resolve_class_id(stem)

    if bbox is not None and poly is not None and len(poly) >= 3:
        poly_norm_values: list[float] = []
        for px, py in poly:
            xn = px / max(w - 1, 1)
            yn = py / max(h - 1, 1)
            poly_norm_values.append(float(xn))
            poly_norm_values.append(float(yn))

        xs = poly_norm_values[0::2]
        ys = poly_norm_values[1::2]
        bbox_norm = [
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            max(xs) - min(xs),
            max(ys) - min(ys),
        ]

        values = [*bbox_norm, *poly_norm_values]
        line = str(class_id) + "".join(f" {v:.6f}" for v in values)

        label_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_save_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        if label_save_path.exists():
            label_save_path.unlink()

    return {"ok": True, "stem": stem}


@app.get("/api/pipeline/config")
def api_pipeline_config():
    return {
        "steps": list(PIPELINE_STEPS),
        "dirs": {k: str(v) for k, v in PIPELINE_DIRS.items()},
        "defaults": {
            "export_bg_prefix": "",
            "training_dataset_yaml": str(cfg.TRAINING_DATASET_YAML),
            "weights_dir": str(cfg.WEIGHTS_DIR),
        },
    }

@app.get("/api/weights")
def api_weights():
    wd = cfg.WEIGHTS_DIR
    if not wd.is_dir():
        return {"weights": []}
    names = sorted(
        {
            p.name
            for p in wd.iterdir()
            if p.is_file() and p.suffix.lower() in (".pt", ".yaml", ".yml")
        }
    )
    return {"weights": names}


@app.get("/api/segment_weights")
def api_segment_weights():
    wd = cfg.WEIGHTS_DIR
    if not wd.is_dir():
        return {"weights": []}

    names: list[str] = []
    for p in wd.rglob("*.pt"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(wd)
            names.append(str(rel).replace("\\", "/"))
        except ValueError:
            names.append(p.name)

    names.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
    return {"weights": names}

@app.get("/api/explore")
def api_explore(path: str = ""):
    target = Path(path)
    
    if not target.exists() or not target.is_dir():
        return {"dirs": [], "files": []}
    
    dirs = []
    files = []
    try:
        for child in target.iterdir():
            if child.is_dir():
                dirs.append({"name": child.name, "path": str(child)})
            else:
                files.append({"name": child.name, "path": str(child)})
    except Exception:
        pass
    
    def natural_sort_key(item):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", item["name"])
        ]
    
    dirs.sort(key=natural_sort_key)
    files.sort(key=natural_sort_key)

    return {"dirs": dirs, "files": files}

@app.get("/api/preview")
def api_preview(path: str):
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    mt, _ = mimetypes.guess_type(target.name)
    media_type = mt or "application/octet-stream"
    
    try:
        if target.suffix.lower() in [".txt", ".yaml", ".json", ".xml", ".log", ".csv"]:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()
            return Response(content=content, media_type="text/plain")
        else:
            with open(target, "rb") as f:
                content = f.read()
            return Response(content=content, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/run")
async def api_pipeline_run(request: Request):
    payload = await request.json()
    step = str(payload.get("step", "")).strip().lower()
    if step not in PIPELINE_STEPS:
        raise HTTPException(status_code=400, detail="Invalid step")
    if step == "train" and not str(payload.get("weights", "")).strip():
        raise HTTPException(status_code=400, detail="weights required for train")

    cmd = _build_docker_command(step, payload)
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "step": step,
            "status": "queued",
            "returncode": None,
            "output": "",
            "command": cmd,
        }
    thread = threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/pipeline/job/{job_id}")
def api_pipeline_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/pipeline/job/{job_id}/stop")
def api_pipeline_job_stop(job_id: str):
    stopped = _stop_job_process(job_id)
    return {"ok": True, "stopped": stopped}


# Web UI
HTML_PAGE = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Annotation Web Edit + SAM Point Assist</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --mask-rgba: rgba(0,255,0,0.43);
    }

    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f7fa;
      color: #333;
    }

    .app {
      display: grid;
      grid-template-columns: 280px 1fr 560px;
      height: 100vh;
    }

    .sidebar {
      border-right: 1px solid #d9dee5;
      background: #fff;
      padding: 12px;
      overflow: auto;
    }

    .main {
      height: 100vh;
      min-width: 0;
      background: #f5f7fa;
    }

    .pipeline-panel {
      border-left: 1px solid #d9dee5;
      background: #fff;
      padding: 16px;
      overflow: auto;
      display: flex;
      flex-direction: column;
    }

    .right-tabs {
      display: flex;
      flex-direction: row;
      gap: 8px;
    }
    .right-tabs button {
      flex: 1;
      padding: 10px 14px;
      text-align: center;
      font-size: 14px;
      font-weight: bold;
      background: #fff;
      border: 1px solid #d9dee5;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .right-tabs button:hover {
      background: #f0f4f8;
    }
    .right-tabs button.active {
      background: #dff1ff;
      border-color: #4a90e2;
      color: #1c5b9e;
    }

    .settings-block {
      border: 1px solid #e6eaf0;
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
      background: #fafbfc;
    }
    .settings-block h4 {
      margin-top: 0;
      margin-bottom: 12px;
      font-size: 15px;
    }

    .gui-container {
      display: grid;
      grid-template-rows: 1fr auto;
      height: 100%;
      min-width: 0;
    }
    
    .center-tab-content {
      padding: 24px;
      background: #fff;
      overflow: auto;
      height: 100%;
      box-sizing: border-box;
    }
    
    .preview-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      box-sizing: border-box;
      padding: 16px 24px;
    }

    .preview-content {
      flex: 1;
      display: flex;
      justify-content: center;
      align-items: center;
      background: #e9edf2;
      border-radius: 8px;
      padding: 16px;
      min-height: 0; 
      min-width: 0;
      overflow: hidden;
      box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
    }

    .preview-content img {
      width: 100%;
      height: 100%;
      object-fit: contain; 
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .preview-content pre {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 16px;
      background: #fff;
      border-radius: 4px;
      overflow: auto;
      font-size: 13px;
      color: #333;
      box-sizing: border-box;
    }

    .field {
      margin-bottom: 12px;
    }
    .field label {
      display: block;
      font-size: 13px;
      color: #475467;
      margin-bottom: 4px;
    }
    .field input, .field select {
      width: 100%;
      box-sizing: border-box;
      padding: 8px;
      border: 1px solid #d9dee5;
      border-radius: 4px;
    }
    
    .field-row {
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }
    .field-row .field {
      margin-bottom: 0;
      flex: 1;
    }

    .field label.checkbox-option {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      height: 36px;
      width: 100%;
      padding: 8px 10px;
      box-sizing: border-box;
      border: 1px solid #d9dee5;
      border-radius: 4px;
      background: #fff;
      color: #344054;
      font-size: 13px;
      cursor: pointer;
      white-space: nowrap;
    }
    .field label.checkbox-option input[type="checkbox"] {
      width: 16px;
      height: 16px;
      margin: 0;
      padding: 0;
      flex: 0 0 auto;
    }
    .checkbox-field label:first-child {
      visibility: visible;
    }
    .field:has(#trainResumeBool),
    .field:has(#trainPretrainedBackboneOnly) {
      display: none;
    }
    .resume-option-row .field:first-child {
      flex: 2 1 0;
    }
    .resume-option-row .checkbox-field,
    .resume-option-row .field:last-child {
      flex: 1 1 0;
    }

    .run-btn {
      width: 100%;
      margin-top: 10px;
      padding: 10px;
      background: #4a90e2;
      color: white;
      border: none;
      border-radius: 6px;
      font-weight: bold;
      cursor: pointer;
    }
    .run-btn:hover { background: #357abd; }

    .train-action-row {
      display: flex;
      gap: 10px;
      margin-top: 10px;
    }
    .train-action-row .run-btn {
      margin-top: 0;
      width: auto;
    }
    .train-run-btn {
      flex: 2 1 0;
    }
    .train-stop-btn {
      flex: 1 1 0;
      background: #9aa3ad;
    }
    .train-stop-btn:hover {
      background: #858f99;
    }
    .train-stop-btn.active {
      background: #e74c3c;
    }
    .train-stop-btn.active:hover {
      background: #c0392b;
    }
    .run-btn:disabled {
      background: #9aa3ad !important;
      color: #eef1f4;
      cursor: not-allowed;
      filter: grayscale(1);
      opacity: 0.75;
    }

    #pipelineLog {
      width: 100%;
      /* Log 창 길이 2.5배 적용 */
      min-height: 450px;
      max-height: 625px;
      font-family: Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      background: #111827;
      color: #e5e7eb;
      border: 1px solid #374151;
      border-radius: 6px;
      padding: 8px;
      box-sizing: border-box;
      overflow: auto;
    }

    .viewer-wrap {
      overflow: auto;
      background: #e9edf2;
      display: flex;
      align-items: flex-start;
      justify-content: flex-start;
      padding: 16px;
      position: relative;
      min-width: 0;
      min-height: 0;
    }

    .canvas-stack {
      position: relative;
      display: inline-block;
      background: #ccc;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
      user-select: none;
      cursor: none;
    }

    canvas {
      position: absolute;
      left: 0;
      top: 0;
      image-rendering: pixelated;
    }

    #baseCanvas { z-index: 1; position: relative; }
    #maskCanvas { z-index: 2; cursor: none; }
    #promptCanvas { z-index: 3; pointer-events: none; }

    #cursorOverlay {
      position: absolute;
      pointer-events: none;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 1px #111;
      border-radius: 9999px;
      transform: translate(-50%, -50%);
      z-index: 10;
      display: none;
      box-sizing: border-box;
      background: transparent;
    }

    .bottom-bar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      border-top: 1px solid #d9dee5;
      background: #ffffff;
      min-height: 72px;
      padding: 10px 12px;
      box-sizing: border-box;
    }
    .bottom-left, .bottom-right {
      display: flex;
      align-items: center;
      gap: 10px;
      box-sizing: border-box;
      min-width: 0;
      flex-wrap: nowrap;
    }
    .bottom-left {
      justify-content: flex-start;
      overflow: hidden;
    }
    .bottom-right {
      justify-content: flex-end;
      margin-left: auto;
      flex-shrink: 0;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 42px;
      padding: 0 10px;
      border-right: 1px solid #e5e7eb;
      box-sizing: border-box;
      flex-shrink: 0;
    }
    .control-group:first-child {
      padding-left: 0;
    }
    .control-group:last-child {
      border-right: 0;
      padding-right: 0;
    }
    .control-group label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      color: #344054;
      font-size: 14px;
    }
    .control-group input[type="range"] {
      width: 150px;
      padding: 0;
    }
    .control-group button {
      min-width: 68px;
      height: 36px;
      padding: 0 12px;
      border: 1px solid #b8c0cc;
      border-radius: 6px;
      background: #ffffff;
      cursor: pointer;
      line-height: 1;
    }
    .mode-group button {
      min-width: 88px;
    }
    .history-group button {
      min-width: 82px;
    }
    .nav-group button {
      min-width: 78px;
    }
    .save-group button {
      min-width: 82px;
      font-weight: bold;
    }
    .info-group {
      border-right: 0;
      padding-right: 0;
    }
    .status-text {
      display: block;
      width: 110px;
      max-width: 110px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .current-info-wrap {
      display: block;
      width: 190px;
      min-width: 0;
      max-width: 190px;
      text-align: right;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    @media (max-width: 1200px) {
      .bottom-bar {
        grid-template-columns: 1fr;
      }
      .bottom-right {
        justify-content: flex-start;
        margin-left: 0;
      }
      .current-info-wrap {
        text-align: left;
      }
    }

    button, select, input[type="range"] {
      padding: 8px 10px;
      font-size: 14px;
    }
    button.active {
      background: #d7ebff;
      border: 1px solid #4a90e2;
    }

    .file-item {
      padding: 8px;
      border: 1px solid #e6eaf0;
      border-radius: 6px;
      margin-bottom: 6px;
      cursor: pointer;
      background: #fff;
    }
    .file-item.active {
      background: #dff1ff;
      border-color: #4a90e2;
      font-weight: bold;
    }

    .left-folder-item {
      background: #fff;
      border: 1px solid #e3e8ef;
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 8px;
      cursor: pointer;
      user-select: none;
    }
    .left-folder-item:hover {
      border-color: #4a90e2;
    }
    .left-folder-item-title {
      font-weight: bold;
      color: #333;
      margin-bottom: 4px;
      font-size: 13px;
    }
    .left-folder-item-path {
      font-size: 11px;
      color: #667085;
      word-break: break-all;
    }
    
    .folder-tree {
      margin-top: 8px;
      padding-left: 12px;
      border-left: 2px solid #e9edf2;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .explorer-item {
      padding: 6px 8px;
      font-size: 13px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .explorer-item:hover {
      background: #f0f4f8;
    }
    .explorer-item.dir {
      color: #333;
      font-weight: bold;
    }
    .explorer-item.file {
      color: #4a90e2;
    }
    .explorer-item.file.active {
      background: #dff1ff;
      color: #1c5b9e;
      font-weight: bold;
    }

    .muted { color: #667085; font-size: 13px; }
    .hint-box {
      margin: 10px 0 12px 0;
      padding: 10px;
      background: #f7f9fc;
      border: 1px solid #e3e8ef;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.45;
      color: #475467;
    }

    .check-result-box {
      padding: 20px;
      border-radius: 8px;
      background: #fff;
      border: 1px solid #e3e8ef;
      margin-top: 16px;
    }
    .check-result-box ul {
      color: #475467;
      margin-bottom: 0;
      padding-left: 20px;
    }
    .check-result-box li {
      margin-bottom: 6px;
    }

    .loading-overlay {
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(17, 24, 39, 0.45);
    }

    .loading-box {
      min-width: 260px;
      padding: 22px 26px;
      border-radius: 10px;
      background: #fff;
      box-shadow: 0 8px 30px rgba(0,0,0,0.20);
      text-align: center;
      color: #333;
    }

    .loading-spinner {
      width: 34px;
      height: 34px;
      margin: 0 auto 12px auto;
      border: 4px solid #e5e7eb;
      border-top-color: #4a90e2;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
<div id="loadingOverlay" class="loading-overlay">
  <div class="loading-box">
    <div class="loading-spinner"></div>
    <div id="loadingText">Loading...</div>
  </div>
</div>
<div class="app">
  <aside class="sidebar" id="leftSidebar">
    
    <div id="guiSidebarContent" style="display: none;">
      <h3 style="margin-top: 0;">Images</h3>
      <div class="muted" id="countInfo"></div>

      <div class="hint-box" style="margin-top: 10px;">
        <b>SAM Point Assist</b><br>
        좌클릭: Positive point + 즉시 SAM 실행<br>
        우클릭: Negative point + 즉시 SAM 실행<br>
        결과는 기존 마스크를 덮어쓰고, 이후 Brush/Eraser로 수정 가능
      </div>
      
      <div id="fileList" style="margin-top: 16px;"></div>
    </div>

    <div id="folderSidebarContent" style="display: none;">
      <h3 style="margin-top: 0;" id="folderSidebarTitle">Folders</h3>
      <div id="leftFolderList"></div>
    </div>

  </aside>

  <main class="main" id="mainArea">
    
    <div id="guiContainer" class="gui-container" style="display: none;">
      <div class="viewer-wrap" id="viewerWrap">
        <div class="canvas-stack" id="canvasStack">
          <canvas id="baseCanvas"></canvas>
          <canvas id="maskCanvas"></canvas>
          <div id="cursorOverlay"></div>
        </div>
      </div>
      <div class="bottom-bar">
        <div class="bottom-left">
          <div class="control-group mode-group">
            <button id="brushBtn" class="active" onclick="setMode('brush')">🖌️ Brush</button>
            <button id="eraseBtn" onclick="setMode('erase')">🧽 Eraser</button>
            <button id="samPointBtn" onclick="setMode('sam_point')">🎯 SAM Point</button>
          </div>
          <div class="control-group brush-group">
            <label>Brush Size <input type="range" id="brushSize" min="1" max="80" value="10" /></label>
          </div>
          <div class="control-group history-group">
            <button onclick="undoMask()">Undo</button>
            <button onclick="clearMask()">Erase All</button>
          </div>
          <div class="control-group zoom-group">
            <label>Zoom <input type="range" id="zoomRange" min="10" max="400" value="100" /></label>
            <span id="status" class="muted status-text">Ready</span>
          </div>
        </div>
        <div class="bottom-right">
          <div class="control-group nav-group">
            <button onclick="prevImage()">◀ Prev</button>
            <button onclick="nextImage()">Next ▶</button>
          </div>
          <div class="control-group save-group">
            <button onclick="saveCurrent()">💾 Save</button>
          </div>
          <div class="control-group info-group">
            <span id="currentInfo" class="muted current-info-wrap"></span>
          </div>
        </div>
      </div>
    </div>

    <div id="previewContainer" class="preview-container" style="display: none;">
      <h2 style="margin-top:0; margin-bottom:16px;">File Preview</h2>
      <div id="previewContent" class="preview-content">
        <span class="muted">Select a file from the left sidebar to preview</span>
      </div>
    </div>

  </main>

  <aside class="pipeline-panel">
    <h3 style="margin-top: 0; color: #1c5b9e;">Pipeline</h3>
    
    <div class="right-tabs">
      <button onclick="switchPipelineTab('input')" id="tabBtn_input">Input</button>
      <button onclick="switchPipelineTab('segment')" id="tabBtn_segment">Segment</button>
      <button onclick="switchPipelineTab('gui')" id="tabBtn_gui">Edit</button>
      <button onclick="switchPipelineTab('export')" id="tabBtn_export">Export</button>
    </div>
    
    <div id="rightSettingsArea">
      <div id="inputSettings" class="settings-block" style="display: none;">
        <h4>Input Settings</h4>
        <div class="muted" style="margin-bottom: 12px;">학습·세그멘테이션용 이미지를 먼저 준비하세요.</div>
        <button class="run-btn" style="margin-top:0;background:#607d8b;" onclick="checkInputStatus()">📂 입력 경로 확인</button>
        <div id="inputCheckResult" class="hint-box" style="margin-top:10px;">
          <div class="muted">아직 확인하지 않았습니다.</div>
        </div>
        <div id="dropZone" class="hint-box" style="border:2px dashed #9db2c8; background:#f8fbff; cursor:pointer;">
          <b>이미지 드래그 앤 드롭</b><br>
          PNG/JPG/JPEG/BMP 파일을 여기에 놓으면 <code>create/input/images</code>로 업로드됩니다.
          <div style="margin-top:8px;">
            <input id="filePicker" type="file" accept=".png,.jpg,.jpeg,.bmp" multiple />
          </div>
        </div>
      </div>
      
      
      <div id="segmentSettings" class="settings-block" style="display: none;">
        <h4>Segment Settings</h4>
        <div class="field">
          <label>Segmentation Weights (.pt)</label>
          <select id="segmentWeights"></select>
        </div>
        <div class="muted" style="margin-bottom: 12px;">weights/ 하위 폴더의 .pt 파일을 자동 탐색합니다.</div>
        <button class="run-btn" onclick="runPipelineStep('segment')">▶ Run Segment</button>
      </div>

      <div id="exportSettings" class="settings-block" style="display: none;">
        <h4>Export Settings</h4>
        <div class="field"><label>Export Prefix (bg)</label><input id="exportBgPrefix" type="text" value="" /></div>
        <button class="run-btn" onclick="runPipelineStep('export')">▶ Run Export</button>
      </div>
      
    </div>

    <!-- 로그 (Editor 하단) -->
    <h4 style="margin-top: 24px; margin-bottom: 8px;">Log</h4>
    <div id="pipelineLog">Waiting...</div>
    
    <div class="extra-tools" style="margin-top: 24px; border-top: 1px solid #d9dee5; padding-top: 16px;">
      <h3 style="margin-top: 0; color: #1c5b9e;">🛠️ Utility</h3>
      
      <div class="right-tabs">
        <button onclick="switchExtraTab('train')" id="extraBtn_train">Training</button>
        <button onclick="switchExtraTab('clean')" id="extraBtn_clean">Clean</button>
      </div>

      <div id="extraSettingsArea">
        <!-- 1. Train (YOLO Training) -->
        <div id="trainSettings" class="settings-block" style="display: none;">
          <h4>YOLO 학습 (Training)</h4>
          <p class="muted" style="margin-top:0;">학습 실행 시 내부적으로 build_split을 먼저 수행해 <code>create/training/dataset.yaml</code>을 생성한 뒤 train합니다.</p>
          <p class="muted">스크래치는 <code>weights/yolo26l-seg.yaml</code> 등, 추가학습은 <code>weights/*.pt</code>를 선택하세요.</p>
          
          <h5 style="margin-bottom:8px;">필수 옵션</h5>
          <div class="field">
            <label>기준 모델 (weights/*.pt 또는 *.yaml)</label>
            <select id="trainWeights"></select>
          </div>
          <div class="field">
            <label>모델 이름 (exp_name)</label>
            <input id="trainName" type="text" placeholder="예: my_exp" />
          </div>
          <div class="field">
            <label>클래스 개수 (num_classes)</label>
            <input id="trainNumClasses" type="number" value="10" min="1" />
          </div>
          <div class="field-row">
              <div class="field">
                <label>Epoch</label>
                <input id="trainEpochs" type="number" value="50" min="1" />
              </div>
              <div class="field">
                <label>Batch size</label>
                <input id="trainBatch" type="number" value="16" min="1" />
              </div>
              <div class="field">
                <label>Image size</label>
                <input id="trainImgsz" type="number" value="640" min="32" step="32" />
              </div>
              <div class="field">
                <label>Val ratio</label>
                <input id="trainValRatio" type="number" value="0.2" min="0.01" max="0.99" step="0.01" />
              </div>
          </div>

          <h5 style="margin-top:16px;">선택 옵션</h5>
          <div class="field-row">
            <div class="field"><label>Learning rate</label><input id="trainLr0" type="number" step="any" placeholder="기본" /></div>
            <div class="field"><label>Patience</label><input id="trainPatience" type="number" placeholder="기본" /></div>
          </div>
          <div class="field-row">
            <div class="field">
              <label>Optimizer</label>
              <select id="trainOptimizer">
                <option value="">auto</option>
                <option value="SGD">SGD</option>
                <option value="AdamW">AdamW</option>
                <option value="Adam">Adam</option>
              </select>
            </div>
            <div class="field"><label>Freeze layers</label><input id="trainFreeze" type="number" min="0" placeholder="0=off" /></div>
          </div>
          <div class="field-row">
            <div class="field">
              <label>Augmentation</label>
              <select id="trainAugment">
                <option value="low">low</option>
                <option value="medium" selected>medium</option>
                <option value="high">high</option>
              </select>
            </div>
            <div class="field"><label>Workers</label><input id="trainWorkers" type="number" min="0" placeholder="기본" /></div>
            <div class="field"><label>Seed</label><input id="trainSeed" type="number" placeholder="기본" /></div>
          </div>
          <div class="field-row">
            <div class="field"><label>Resume</label><input id="trainResumePath" type="text" placeholder="체크포인트 경로 또는 비움" /></div>
            <div class="field checkbox-field">
              <label>Resume auto</label>
              <label class="checkbox-option"><input type="checkbox" id="trainResumeAutoProxy" /></label>
            </div>
            <div class="field checkbox-field">
              <label>Pretrained backbone</label>
              <label class="checkbox-option"><input type="checkbox" id="trainPretrainedBackboneOnlyProxy" /></label>
            </div>
            <div class="field">
              <label>Cache</label>
              <select id="trainCache">
                <option value="">(기본)</option>
                <option value="true">true</option>
                <option value="false">false</option>
                <option value="ram">ram</option>
                <option value="disk">disk</option>
              </select>
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label>Mixed precision (fp16)</label>
              <select id="trainAmp">
                <option value="">(기본)</option>
                <option value="true">on</option>
                <option value="false">off</option>
              </select>
            </div>
            <div class="field">
              <label>Cosine LR</label>
              <select id="trainCosLr">
                <option value="">(기본)</option>
                <option value="true">on</option>
                <option value="false">off</option>
              </select>
            </div>
            <div class="field"><label><input type="checkbox" id="trainResumeBool" /> Resume (자동 이어하기)</label></div>
          </div>
          <div class="field-row">
            <div class="field"><label>Weight decay</label><input id="trainWd" type="number" step="any" placeholder="기본" /></div>
            <div class="field"><label>Dropout</label><input id="trainDropout" type="number" step="any" placeholder="기본" /></div>
            <div class="field"><label>Save period</label><input id="trainSavePeriod" type="number" placeholder="기본" /></div>
          </div>
          <div class="field">
            <label><input type="checkbox" id="trainPretrainedBackboneOnly" /> Pretrained backbone only</label>
          </div>
          <div class="train-action-row">
            <button id="trainRunBtn" class="run-btn train-run-btn" style="background:#3498db;" onclick="runTrain()">▶ 학습 실행</button>
            <button id="trainStopBtn" class="run-btn train-stop-btn" onclick="stopTraining()" disabled>■ 학습 중단</button>
          </div>
        </div>

        <!-- 2. Clean -->
        <div id="cleanSettings" class="settings-block" style="display: none;">
          <h4>작업 폴더 정리 (Clean)</h4>
          <div class="field">
            <label>삭제 대상</label>
            <select id="cleanMode">
              <option value="all">all - 전체 작업 폴더 초기화</option>
              <option value="input">input - 업로드한 입력 이미지 삭제</option>
              <option value="output1">output1 - 자동 세그멘테이션 결과 삭제</option>
              <option value="output2">output2 - Edit 수정 결과 삭제</option>
              <option value="dataset">dataset - 최종 생성 Dataset 삭제</option>
              <option value="training">training - 학습 결과 및 가중치 삭제</option>
            </select>
          </div>
          <button class="run-btn" style="background:#e74c3c;" onclick="runClean()">🗑️ 선택 폴더 초기화</button>
        </div>
      </div>
    </div>

  </aside>
</div>

<script>
let items = [];
let currentIndex = 0;
let currentStem = null;
let mode = "brush";
let drawing = false;
let zoom = 1.0;
let undoStack = [];
let isDirty = false;
let isLoading = false;
let isSamRunning = false;
let samPoints = [];
let pipelineConfig = null;
let activePipelineTab = ""; 
let currentJobId = null;
let currentTrainingJobId = null;

const MASK_DRAW_COLOR = "rgba(0,255,0,0.43)";
const MASK_DRAW_ALPHA_THRESHOLD = 20;

const baseCanvas = document.getElementById("baseCanvas");
const maskCanvas = document.getElementById("maskCanvas");
const baseCtx = baseCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d");

const brushSizeEl = document.getElementById("brushSize");
const zoomRangeEl = document.getElementById("zoomRange");
const statusEl = document.getElementById("status");
const fileListEl = document.getElementById("fileList");
const currentInfoEl = document.getElementById("currentInfo");
const countInfoEl = document.getElementById("countInfo");
const viewerWrapEl = document.getElementById("viewerWrap");
const canvasStackEl = document.getElementById("canvasStack");
const cursorOverlayEl = document.getElementById("cursorOverlay");
const pipelineLogEl = document.getElementById("pipelineLog");
const loadingOverlayEl = document.getElementById("loadingOverlay");
const loadingTextEl = document.getElementById("loadingText");
const trainRunBtnEl = document.getElementById("trainRunBtn");
const trainStopBtnEl = document.getElementById("trainStopBtn");

const stepFolderMap = {
    input: [
        { label: "create/input/images", key: "input_images" }
    ],
    segment: [
        { label: "create/output_1/images/", key: "segment_images" },
        { label: "create/output_1/labels/", key: "segment_labels" },
        { label: "create/output_1/masks/", key: "segment_masks" }
    ],
    export: [
        { label: "create/dataset/images", key: "export_images" },
        { label: "create/dataset/labels", key: "export_labels" },
        { label: "create/dataset/masks", key: "export_masks" }
    ]
};

function setStatus(msg) { statusEl.textContent = msg; }
function setPipelineLog(msg) { pipelineLogEl.textContent = msg || "No logs"; }

function setTrainingButtonsRunning(isRunning) {
    if (trainRunBtnEl) trainRunBtnEl.disabled = isRunning;
    if (trainStopBtnEl) {
        trainStopBtnEl.disabled = !isRunning;
        trainStopBtnEl.classList.toggle("active", isRunning);
    }
}

function showLoading(msg = "Loading...") {
    loadingTextEl.textContent = msg;
    loadingOverlayEl.style.display = "flex";
}

function hideLoading() {
    loadingOverlayEl.style.display = "none";
}

function formatPipelineOutput(raw) {
    const lines = String(raw || "").split("\n");
    const kept = [];
    let yoloProgress = null;

    for (const line of lines) {
        const m = line.match(/YOLO batches:\s*(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/);
        if (m) {
            yoloProgress = {
                percent: Number(m[1]),
                done: Number(m[2]),
                total: Number(m[3]),
            };
            continue;
        }
        kept.push(line);
    }

    if (yoloProgress) {
        const barLen = 20;
        const fill = Math.max(0, Math.min(barLen, Math.round((yoloProgress.percent / 100) * barLen)));
        const bar = "█".repeat(fill) + "░".repeat(barLen - fill);
        kept.push(`YOLO batches progress: [${bar}] ${yoloProgress.percent}% (${yoloProgress.done}/${yoloProgress.total})`);
    }
    return kept.join("\n").trim();
}

formatPipelineOutput = function(raw) {
    const ansiPattern = /\x1b\[[0-?]*[ -/]*[@-~]/g;
    const lines = String(raw || "").replace(/\r/g, "\n").split("\n");
    const kept = [];
    let trainProgress = null;
    let valProgress = null;
    let yoloProgress = null;
    let trainProgressIndex = -1;
    let valProgressIndex = -1;
    let yoloProgressIndex = -1;
    let trainHeaderKept = false;
    let metricsHeaderKept = false;
    let currentEpoch = "";
    let currentLosses = {};

    function makeProgressLine(label, percent, done, total, suffix = "") {
        const barLen = 24;
        const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
        const fill = Math.max(0, Math.min(barLen, Math.round((safePercent / 100) * barLen)));
        const bar = "#".repeat(fill) + "-".repeat(barLen - fill);
        const count = done && total ? ` (${done}/${total})` : "";
        return `${label}: [${bar}] ${safePercent}%${count}${suffix ? " " + suffix.trim() : ""}`;
    }

    function makeMetricsRow(values) {
        const widths = [6, 9, 9, 9, 9, 11];
        const align = (value, width, left = false) => {
            const text = String(value || "");
            return left ? text.padEnd(width, " ") : text.padStart(width, " ");
        };
        return [
            align(values[0], widths[0]),
            align(values[1], widths[1]),
            align(values[2], widths[2]),
            align(values[3], widths[3]),
            align(values[4], widths[4]),
            align(values[5], widths[5]),
        ].join(" ");
    }

    for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const cleanLine = line.replace(ansiPattern, "").trimEnd();
        if (!cleanLine.trim()) {
            continue;
        }

        if (/^\s*Epoch\s+GPU_mem\s+box_loss\s+seg_loss\s+cls_loss\s+dfl_loss\s+sem_loss\s+Instances\s+Size\s*$/.test(cleanLine)) {
            trainHeaderKept = true;
            continue;
        }

        const trainMatch = cleanLine.match(/^\s*(\d+)\/(\d+).*?:\s*(\d+)%\s+.*?\s(\d+)\/(\d+)\s*(.*)$/);
        if (trainMatch) {
            currentEpoch = trainMatch[1];
            const beforeProgress = cleanLine.split(":")[0].trim().split(/\s+/);
            if (beforeProgress.length >= 7) {
                currentLosses = {
                    seg: beforeProgress[3],
                    cls: beforeProgress[4],
                    sem: beforeProgress[6],
                };
            }
            trainProgress = makeProgressLine(
                `Train epoch ${trainMatch[1]}/${trainMatch[2]}`,
                trainMatch[3],
                trainMatch[4],
                trainMatch[5],
                trainMatch[6]
            );
            trainProgressIndex = i;
            continue;
        }

        const valMatch = cleanLine.match(/^\s*Class\s+Images\s+Instances.*?:\s*(\d+)%\s+.*?\s(\d+)\/(\d+)\s*(.*)$/);
        if (valMatch) {
            valProgress = makeProgressLine("Validation", valMatch[1], valMatch[2], valMatch[3], valMatch[4]);
            valProgressIndex = i;
            continue;
        }

        const yoloBatchMatch = cleanLine.match(/^YOLO batches:\s*(\d+)%\|.*?\|\s*(\d+)\/(\d+)\s*(.*)$/);
        if (yoloBatchMatch) {
            yoloProgress = makeProgressLine(
                "YOLO batches",
                yoloBatchMatch[1],
                yoloBatchMatch[2],
                yoloBatchMatch[3],
                yoloBatchMatch[4]
            );
            yoloProgressIndex = i;
            continue;
        }

        if (/^\s*Class\s+Images\s+Instances\s+Box\(P\s+R\s+mAP50\s+mAP50-95\)\s+Mask\(P\s+R\s+mAP50\s+mAP50-95\)\s*$/.test(cleanLine)) {
            continue;
        }

        if (/^\s*all\s+/.test(cleanLine)) {
            const parts = cleanLine.trim().split(/\s+/);
            if (!metricsHeaderKept) {
                kept.push(makeMetricsRow(["epoch", "seg_loss", "cls_loss", "sem_loss", "mask_m50", "mask_m50-95"]));
                metricsHeaderKept = true;
            }
            kept.push(makeMetricsRow([
                currentEpoch || "-",
                currentLosses.seg || "-",
                currentLosses.cls || "-",
                currentLosses.sem || "-",
                parts[9] || "-",
                parts[10] || "-",
            ]));
            continue;
        }

        kept.push(cleanLine.trimEnd().replace(/\s{2,}/g, " "));
    }

    while (kept.length && kept[kept.length - 1] === "") kept.pop();
    if (trainProgressIndex > valProgressIndex) valProgress = null;
    if (trainProgress) kept.push(trainProgress);
    if (valProgress) kept.push(valProgress);
    if (yoloProgress && yoloProgressIndex > trainProgressIndex && yoloProgressIndex > valProgressIndex) {
        kept.push(yoloProgress);
    }
    return kept.join("\n").trim();
};

async function checkInputStatus() {
    try {
        const res = await fetch("/api/input/status");
        const data = await res.json();
        const el = document.getElementById("inputCheckResult");
        if (!res.ok) throw new Error(data.detail || "status fetch failed");
        if ((data.count || 0) > 0) {
            el.innerHTML = `<b>입력 확인됨</b><br>경로: <code>${data.dir}</code><br>파일 수: <b>${data.count}</b>`;
        } else {
            el.innerHTML = `<b>입력 파일이 없습니다</b><br>경로: <code>${data.dir}</code><br>아래 드래그앤드롭으로 이미지를 추가하세요.`;
        }
    } catch (e) {
        document.getElementById("inputCheckResult").innerHTML = `<span style="color:red;">입력 경로 확인 실패: ${String(e)}</span>`;
    }
}

async function uploadInputFiles(fileList) {
    if (!fileList || fileList.length === 0) return;
    const fd = new FormData();
    const files = prepareInputFilesForUpload(fileList);
    if (!files) return;
    for (const f of files) fd.append("files", f, f.name);
    setStatus("Uploading input files...");
    try {
        const res = await fetch("/api/input/upload", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "upload failed");
        setStatus(`Uploaded: ${data.saved.length}, Skipped: ${data.skipped.length}`);
        await checkInputStatus();
    } catch (e) {
        setStatus("Upload failed");
        alert(String(e));
    }
}

function splitFileName(name) {
    const dot = name.lastIndexOf(".");
    if (dot <= 0) return { stem: name, ext: "" };
    return { stem: name.slice(0, dot), ext: name.slice(dot) };
}

function startsWithClassNumber(name) {
    const { stem } = splitFileName(name);
    const first = stem.split("_")[0];
    return /^\d+$/.test(first);
}

function defaultClassNumberFromName(name) {
    const { stem } = splitFileName(name);
    const match = stem.match(/^class_(\d+)(?:_|$)/i);
    return match ? match[1] : "";
}

function buildClassNumberedName(name, classNumber) {
    const { stem, ext } = splitFileName(name);
    const classPrefix = new RegExp(`^class_${classNumber}_?`, "i");
    const cleanStem = stem.replace(classPrefix, "") || "image";
    return `${classNumber}_${cleanStem}${ext}`;
}

function promptClassNumberForUpload(files) {
    const firstFile = files[0];
    const defaultValue = defaultClassNumberFromName(firstFile.name);
    const message = `${files.length}개 파일의 파일명이 클래스 번호로 시작하지 않습니다.\n전체에 적용할 class_number를 입력하세요.\n첫 파일: ${firstFile.name}`;
    while (true) {
        const value = prompt(message, defaultValue);
        if (value === null) return null;
        const trimmed = value.trim();
        if (/^\d+$/.test(trimmed)) return trimmed;
        alert("class_number는 0 이상의 정수로 입력하세요.");
    }
}

function prepareInputFilesForUpload(fileList) {
    const sourceFiles = Array.from(fileList);
    const needsClassNumber = sourceFiles.filter(file => !startsWithClassNumber(file.name));
    let classNumber = null;
    if (needsClassNumber.length > 0) {
        classNumber = promptClassNumberForUpload(needsClassNumber);
        if (classNumber === null) {
            setStatus("Upload cancelled");
            return null;
        }
    }

    const prepared = [];
    for (const file of sourceFiles) {
        if (startsWithClassNumber(file.name)) {
            prepared.push(file);
            continue;
        }

        const newName = buildClassNumberedName(file.name, classNumber);
        prepared.push(new File([file], newName, {
            type: file.type,
            lastModified: file.lastModified,
        }));
    }
    return prepared;
}

async function loadWeightsList() {
    try {
        const res = await fetch("/api/weights");
        const data = await res.json();
        const sel = document.getElementById("trainWeights");
        sel.innerHTML = "";
        (data.weights || []).forEach(w => {
            const opt = document.createElement("option");
            opt.value = w;
            opt.textContent = w;
            sel.appendChild(opt);
        });
        if (sel.options.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "(weights 폴더에 .pt 없음)";
            sel.appendChild(opt);
        }
    } catch (e) {
        console.error("Failed to load weights", e);
    }
}

async function loadSegmentWeights() {
    try {
        const res = await fetch("/api/segment_weights");
        const data = await res.json();
        const sel = document.getElementById("segmentWeights");
        sel.innerHTML = "";

        const defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = "기본값 사용 (config.py YOLO_WEIGHT)";
        sel.appendChild(defaultOpt);

        (data.weights || []).forEach(w => {
            const opt = document.createElement("option");
            opt.value = w;
            opt.textContent = w;
            sel.appendChild(opt);
        });

        if (sel.options.length === 1 && !(data.weights || []).length) {
            defaultOpt.textContent = "기본값 사용 (.pt 파일 없음)";
        }
    } catch (e) {
        console.error("Failed to load segment weights", e);
    }
}

async function previewFile(filePath, fileName) {
    const container = document.getElementById("previewContent");
    container.innerHTML = '<span class="muted">Loading preview...</span>';
    
    try {
        const isText = fileName.match(/\.(txt|yaml|json|xml|log|csv)$/i);
        const url = `/api/preview?path=${encodeURIComponent(filePath)}`;
        
        if (isText) {
            const res = await fetch(url);
            if (!res.ok) throw new Error("Failed to load file text");
            const text = await res.text();
            container.innerHTML = `<pre>${text}</pre>`;
        } else {
            container.innerHTML = `<img src="${url}" />`;
        }
    } catch (err) {
        container.innerHTML = `<span style="color:red;">Preview not available</span>`;
    }
}

async function toggleSubFolder(path, parentElem) {
    let tree = parentElem.querySelector(':scope > .folder-tree');
    if (tree) {
        tree.remove(); 
        return;
    }
    
    tree = document.createElement('div');
    tree.className = 'folder-tree';
    tree.innerHTML = '<div class="muted" style="padding:4px;">Loading...</div>';
    parentElem.appendChild(tree);
    
    try {
        const res = await fetch(`/api/explore?path=${encodeURIComponent(path)}`);
        const data = await res.json();
        
        tree.innerHTML = '';
        for (const d of data.dirs) {
            const row = document.createElement('div');
            row.innerHTML = `<span>📁 ${d.name}</span>`;
            row.className = 'explorer-item dir';
            row.onclick = (e) => { e.stopPropagation(); toggleSubFolder(d.path, row); };
            tree.appendChild(row);
        }
        for (const f of data.files) {
            const row = document.createElement('div');
            row.innerHTML = `<span>📄 ${f.name}</span>`;
            row.className = 'explorer-item file';
            row.onclick = (e) => { 
                e.stopPropagation(); 
                document.querySelectorAll('.explorer-item.file').forEach(el => el.classList.remove('active'));
                row.classList.add('active');
                previewFile(f.path, f.name); 
            }; 
            tree.appendChild(row);
        }
        if (!data.dirs.length && !data.files.length) {
            tree.innerHTML = '<div class="muted" style="padding:4px;">Empty</div>';
        }
    } catch(e) {
        tree.innerHTML = '<div style="color:red; padding:4px;">Error loading folder</div>';
    }
}

function renderLeftFolders(step) {
    const container = document.getElementById("leftFolderList");
    container.innerHTML = "";
    const items = stepFolderMap[step] || [];
    
    items.forEach(item => {
        const pathInfo = pipelineConfig?.dirs[item.key];
        if (!pathInfo) return;
        
        const block = document.createElement("div");
        block.className = "left-folder-item";
        
        const title = document.createElement("div");
        title.className = "left-folder-item-title";
        title.innerHTML = `📁 ${item.label}`;
        
        const pathEl = document.createElement("div");
        pathEl.className = "left-folder-item-path";
        pathEl.textContent = pathInfo;
        
        block.appendChild(title);
        block.appendChild(pathEl);
        
        block.onclick = (e) => {
            e.stopPropagation();
            toggleSubFolder(pathInfo, block);
        };
        
        container.appendChild(block);
    });
}

async function moveRemaining(btnEvent = null) {
  if (isLoading) return;
  isLoading = true;
  setStatus("Copying unedited files...");
  
  const btn = btnEvent ? btnEvent.target : document.getElementById("btnMoveRemaining");
  if (btn) {
      btn.disabled = true;
      btn.textContent = "Copying...";
  }

  try {
    const res = await fetch("/api/move_remaining", { method: "POST" });
    const data = await res.json();
    if ((data.moved || 0) > 0) {
      setStatus(`Copied ${data.moved} items`);
      alert(`성공적으로 처리되었습니다.\nCopied unedited items to output_2: ${data.moved}`);
    } else {
      setStatus("No output_1 items to copy");
    }
  } catch (err) {
    console.error(err);
    setStatus("Copy failed");
    alert("Copy failed");
  } finally {
    isLoading = false;
    if (btn) {
        btn.disabled = false;
        btn.textContent = "Copy Unedited to Output2";
    }
  }
}

async function move_remaining_silent() {
  try {
    const res = await fetch("/api/move_remaining", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "copy failed");
    setStatus(`Copied ${data.moved} unedited items to output_2`);
    return true;
  } catch (err) {
    console.error(err);
    setStatus("Copy unedited failed");
    alert(`Copy unedited failed: ${String(err)}`);
    return false;
  }
}

async function getMoveRemainingPendingCount() {
  const res = await fetch(cacheBust("/api/move_remaining/status"), { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "status check failed");
  return data.pending || 0;
}

async function switchPipelineTab(step) {
    if (isLoading) return;

    if (activePipelineTab === 'gui' && step !== 'gui' && step !== '') {
        let pendingMoveCount = 0;
        try {
            pendingMoveCount = await getMoveRemainingPendingCount();
        } catch (err) {
            console.error(err);
            setStatus("Copy status check failed");
            alert(`Copy status check failed: ${String(err)}`);
            return;
        }

        if (pendingMoveCount > 0 || isDirty) {
            const msg = [
            "Edit 화면을 벗어나기 전에 확인이 필요합니다.",
            "",
            "확인: 편집 안 한 이미지를 output_2로 자동 복사한 뒤 이동",
            "취소: 복사하지 않고 바로 다음 탭으로 이동",
            ].join("\n");

            if (confirm(msg)) {
                isLoading = true;
                showLoading("편집 안 된 파일을 output_2로 복사하는 중...");
                const ok = await move_remaining_silent();
                hideLoading();
                isLoading = false;

                if (!ok) {
                    return;
                }
            }
        }
    }

    activePipelineTab = step;
    const tabs = ["input", "segment", "gui", "export"];
  
    tabs.forEach(t => {
        const btn = document.getElementById("tabBtn_" + t);
        if (btn) btn.classList.toggle("active", t === step);
        
        const settingsBlock = document.getElementById(t + "Settings");
        if (settingsBlock) {
            settingsBlock.style.display = (t === step) ? "block" : "none";
        }
    });

    document.getElementById("guiContainer").style.display = (step === "gui") ? "grid" : "none";
    document.getElementById("previewContainer").style.display = (step !== "gui") ? "flex" : "none";

    document.getElementById("guiSidebarContent").style.display = (step === "gui") ? "block" : "none";
    document.getElementById("folderSidebarContent").style.display = (step !== "gui") ? "block" : "none";

    if (step === "gui") {
        await loadList({ forceReload: true, autoFit: true });
        if (baseCanvas.width > 0) fitZoomToViewerHeight();
    } else {
        document.getElementById("previewContent").innerHTML = '<span class="muted">Select a file from the left sidebar to preview</span>';
        document.getElementById("folderSidebarTitle").textContent = step.toUpperCase() + " Paths";
        renderLeftFolders(step); 
    }
}

function switchExtraTab(tab) {
    const extraTabs = ["train", "clean"];
    extraTabs.forEach(t => {
        const btn = document.getElementById("extraBtn_" + t);
        if (btn) btn.classList.toggle("active", t === tab);
        
        const settingsBlock = document.getElementById(t + "Settings");
        if (settingsBlock) {
            settingsBlock.style.display = (t === tab) ? "block" : "none";
        }
    });
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/api/pipeline/job/${encodeURIComponent(jobId)}`);
    const data = await res.json();
    if (!res.ok) {
      setPipelineLog(data.detail || "Job poll failed");
      return false;
    }
    const command = Array.isArray(data.command) ? data.command.join(" ") : "";
    const output = formatPipelineOutput(data.output || "");
    setPipelineLog(`[${data.status}] ${command}\n\n${output}`);
    
    if (data.status === "done") {
      setStatus(`Pipeline ${data.step}: ${data.status}`);
      return true;
    } else if (data.status === "stopped") {
      setStatus(`Pipeline ${data.step}: stopped`);
      return false;
    } else if (data.status === "failed") {
      setStatus(`Pipeline ${data.step}: ${data.status}`);
      return false;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function runPipelineStep(step) {
  const payload = { step };
  if (step === "export") {
    payload.bg_prefix = document.getElementById("exportBgPrefix").value;
  }

  if (step === "segment") {
    payload.weights = document.getElementById("segmentWeights").value;
  }

  setStatus(`Starting ${step}...`);
  try {
    const res = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to run pipeline step");
    currentJobId = data.job_id;
    setPipelineLog(`[queued] ${data.command}`);

    const ok = await pollJob(currentJobId);

    if (ok && step === "segment") {
      currentStem = null;
      currentIndex = 0;

      if (activePipelineTab === "gui") {
        await loadList({ forceReload: true, autoFit: true });
      } else {
        renderLeftFolders(activePipelineTab);
      }
    }
  } catch (err) {
    setStatus(`Failed to start ${step}`);
    alert(String(err));
  }
}

async function runTrain() {
    const weights = document.getElementById("trainWeights").value;
    const modelName = document.getElementById("trainName").value.trim();
    const numClassesRaw = document.getElementById("trainNumClasses").value.trim();
    if (!weights || !modelName || !numClassesRaw) {
        alert("가중치(.pt / .yaml), 모델 이름(exp_name), 클래스 개수(num_classes)는 필수입니다.");
        return;
    }
    const numClasses = parseInt(numClassesRaw, 10);
    if (!(numClasses > 0)) {
        alert("num_classes는 1 이상의 정수여야 합니다.");
        return;
    }

    const valRatioRaw = document.getElementById("trainValRatio").value.trim();
    if (valRatioRaw === "") {
        alert("val_ratio는 필수입니다.");
        return;
    }
    const valRatio = parseFloat(valRatioRaw);
    if (!(valRatio > 0.0 && valRatio < 1.0)) {
        alert("val_ratio는 0~1 사이 값이어야 합니다.");
        return;
    }

    const dataYaml = (pipelineConfig && pipelineConfig.defaults && pipelineConfig.defaults.training_dataset_yaml) || "";
    const weightsDir = (pipelineConfig && pipelineConfig.defaults && pipelineConfig.defaults.weights_dir) || "";

    const trainPayload = {
        step: "train",
        weights,
        model_name: modelName,
        epochs: parseInt(document.getElementById("trainEpochs").value || "100", 10),
        batch: parseInt(document.getElementById("trainBatch").value || "16", 10),
        imgsz: parseInt(document.getElementById("trainImgsz").value || "640", 10),
        num_classes: numClasses,
        val_ratio: valRatio,
    };
    if (dataYaml) trainPayload.data = dataYaml;
    if (weightsDir) trainPayload.project = weightsDir;

    const lr0 = document.getElementById("trainLr0").value.trim();
    if (lr0 !== "") trainPayload.lr0 = parseFloat(lr0);
    const patience = document.getElementById("trainPatience").value.trim();
    if (patience !== "") trainPayload.patience = parseInt(patience, 10);
    const opt = document.getElementById("trainOptimizer").value;
    if (opt) trainPayload.optimizer = opt;
    const freeze = document.getElementById("trainFreeze").value.trim();
    if (freeze !== "") trainPayload.freeze = parseInt(freeze, 10);
    trainPayload.augment = document.getElementById("trainAugment").value || "medium";
    const workers = document.getElementById("trainWorkers").value.trim();
    if (workers !== "") trainPayload.workers = parseInt(workers, 10);
    const seed = document.getElementById("trainSeed").value.trim();
    if (seed !== "") trainPayload.seed = parseInt(seed, 10);

    const resumePath = document.getElementById("trainResumePath").value.trim();
    const resumeBool = document.getElementById("trainResumeBool").checked;
    if (resumePath) {
        trainPayload.resume = resumePath;
    } else if (resumeBool) {
        trainPayload.resume = true;
    }

    const cache = document.getElementById("trainCache").value;
    if (cache) trainPayload.cache = cache;

    const ampSel = document.getElementById("trainAmp").value;
    if (ampSel === "true") trainPayload.amp = true;
    else if (ampSel === "false") trainPayload.amp = false;

    const cosSel = document.getElementById("trainCosLr").value;
    if (cosSel === "true") trainPayload.cos_lr = true;
    else if (cosSel === "false") trainPayload.cos_lr = false;

    const wd = document.getElementById("trainWd").value.trim();
    if (wd !== "") trainPayload.weight_decay = parseFloat(wd);
    const dropout = document.getElementById("trainDropout").value.trim();
    if (dropout !== "") trainPayload.dropout = parseFloat(dropout);
    const savePeriod = document.getElementById("trainSavePeriod").value.trim();
    if (savePeriod !== "") trainPayload.save_period = parseInt(savePeriod, 10);

    if (document.getElementById("trainPretrainedBackboneOnly").checked) {
        trainPayload.pretrained_backbone_only = true;
    }

    setStatus("Starting training...");
    try {
        const trainRes = await fetch("/api/pipeline/run", {
            method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(trainPayload)
        });
        const trainData = await trainRes.json();
        if (!trainRes.ok) throw new Error(trainData.detail);
        const trainJobId = trainData.job_id;
        currentJobId = trainJobId;
        currentTrainingJobId = trainJobId;
        setTrainingButtonsRunning(true);
        setPipelineLog(`[queued] ${trainData.command}`);
        const ok = await pollJob(trainJobId);
        if (currentTrainingJobId === trainJobId) {
            currentTrainingJobId = null;
            setTrainingButtonsRunning(false);
        }

        if (ok) {
            await loadWeightsList();
            await loadSegmentWeights();
            if (activePipelineTab === "segment") {
                await loadSegmentWeights();
            }
            setStatus("Training completed. Weights list refreshed.");
        }
    } catch(e) {
        setTrainingButtonsRunning(false);
        currentTrainingJobId = null;
        setStatus("Train failed");
        alert(e);
    }
}

async function stopTraining() {
    if (!currentTrainingJobId) {
        alert("중단할 학습 작업이 없습니다.");
        return;
    }
    setStatus("Stopping training...");
    try {
        const res = await fetch(`/api/pipeline/job/${encodeURIComponent(currentTrainingJobId)}/stop`, {
            method: "POST"
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to stop training");
        setStatus(data.stopped ? "Training stop requested." : "No running training job.");
        if (!data.stopped) {
            currentTrainingJobId = null;
            setTrainingButtonsRunning(false);
        }
    } catch (e) {
        setStatus("Failed to stop training");
        alert(e);
    }
}

async function runClean() {
    if(!confirm("정말로 작업 폴더를 정리하시겠습니까?\n이 작업은 되돌릴 수 없습니다.")) return;
    
    const cleanMode = document.getElementById("cleanMode").value;
    setStatus("Cleaning workspace...");
    const payload = {
        step: "clean",
        clean_mode: cleanMode
    };
    
    try {
        const res = await fetch("/api/pipeline/run", {
            method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);

        const ok = await pollJob(data.job_id);

        if (ok) {
            if (
                cleanMode === "all" ||
                cleanMode === "input" ||
                cleanMode === "output1" ||
                cleanMode === "output2"
            ) {
                currentStem = null;
                currentIndex = 0;

                if (activePipelineTab === "gui") {
                    await loadList({ forceReload: true, autoFit: true });
                } else {
                    renderLeftFolders(activePipelineTab);
                }
            }

            setStatus("Workspace clean completed.");
        }
    } catch(e) {
        setStatus("Clean failed");
        alert(e);
    }
}

async function loadPipelineConfig() {
  const res = await fetch("/api/pipeline/config");
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load pipeline config");
  pipelineConfig = data;
  
  document.getElementById("exportBgPrefix").value = data.defaults.export_bg_prefix;
}

function setDirty(v = true) {
  isDirty = v;
  if (currentStem) {
    const suffix = isDirty ? " *unsaved" : "";
    const base = currentInfoEl.dataset.baseText || currentInfoEl.textContent || "";
    currentInfoEl.textContent = base + suffix;
  }
}

function setMode(newMode) {
  mode = newMode;
  document.getElementById("brushBtn").classList.toggle("active", mode === "brush");
  document.getElementById("eraseBtn").classList.toggle("active", mode === "erase");
  document.getElementById("samPointBtn").classList.toggle("active", mode === "sam_point");

  if (mode === "brush") setStatus("Brush mode");
  else if (mode === "erase") setStatus("Eraser mode");
  else setStatus("SAM point mode (left=positive, right=negative, auto-run)");
  updateCursor();
}

function pushUndo() {
  try {
    undoStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
    if (undoStack.length > 30) undoStack.shift();
  } catch (e) {}
}

function undoMask() {
  if (undoStack.length === 0) return;
  const prev = undoStack.pop();
  maskCtx.putImageData(prev, 0, 0);
  setDirty(true);
  setStatus("Undo");
}

function clearMask() {
  pushUndo();
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  setDirty(true);
  setStatus("Mask cleared");
}

function getPointerPos(evt) {
  const rect = maskCanvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) * (maskCanvas.width / rect.width);
  const y = (evt.clientY - rect.top) * (maskCanvas.height / rect.height);
  return { x, y };
}

function setBrushStyle(ctx) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (mode === "brush") {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = MASK_DRAW_COLOR;
    ctx.fillStyle = MASK_DRAW_COLOR;
  } else {
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
    ctx.fillStyle = "rgba(0,0,0,1)";
  }
}

function drawDot(x, y) {
  const r = parseInt(brushSizeEl.value, 10);
  setBrushStyle(maskCtx);
  maskCtx.beginPath();
  maskCtx.arc(x, y, r, 0, Math.PI * 2);
  maskCtx.fill();
}

function drawLine(x0, y0, x1, y1) {
  const r = parseInt(brushSizeEl.value, 10);
  setBrushStyle(maskCtx);
  maskCtx.lineWidth = Math.max(1, r * 2);
  maskCtx.beginPath();
  maskCtx.moveTo(x0, y0);
  maskCtx.lineTo(x1, y1);
  maskCtx.stroke();
  maskCtx.beginPath();
  maskCtx.arc(x1, y1, r, 0, Math.PI * 2);
  maskCtx.fill();
}

function updateCursor(evt = null) {
  if (mode === "sam_point") {
    cursorOverlayEl.style.width = "14px";
    cursorOverlayEl.style.height = "14px";
  } else {
    const diameter = parseInt(brushSizeEl.value, 10) * 2 * zoom;
    cursorOverlayEl.style.width = `${Math.max(2, diameter)}px`;
    cursorOverlayEl.style.height = `${Math.max(2, diameter)}px`;
  }

  if (!evt) return;
  const rect = canvasStackEl.getBoundingClientRect();
  const x = evt.clientX - rect.left;
  const y = evt.clientY - rect.top;
  cursorOverlayEl.style.left = `${x}px`;
  cursorOverlayEl.style.top = `${y}px`;
}

maskCanvas.addEventListener("contextmenu", (e) => {
  if (mode === "sam_point") e.preventDefault();
});

maskCanvas.addEventListener("mouseenter", () => {
  cursorOverlayEl.style.display = "block";
  updateCursor();
});

maskCanvas.addEventListener("mouseleave", () => {
  cursorOverlayEl.style.display = "none";
});

maskCanvas.addEventListener("mousedown", (evt) => {
  if (!currentStem || isLoading) return;

  const p = getPointerPos(evt);
  if (mode === "sam_point") {
    const isRight = evt.button === 2;
    samPoints.push({ x: p.x, y: p.y, label: isRight ? 0 : 1 });
    updateCursor(evt);
    runSam(true);
    return;
  }

  drawing = true;
  pushUndo();
  maskCanvas.dataset.lastX = String(p.x);
  maskCanvas.dataset.lastY = String(p.y);
  drawDot(p.x, p.y);
  setDirty(true);
  updateCursor(evt);
});

maskCanvas.addEventListener("mousemove", (evt) => {
  updateCursor(evt);
  if (!drawing) return;
  const p = getPointerPos(evt);
  const lastX = parseFloat(maskCanvas.dataset.lastX || "0");
  const lastY = parseFloat(maskCanvas.dataset.lastY || "0");
  drawLine(lastX, lastY, p.x, p.y);
  maskCanvas.dataset.lastX = String(p.x);
  maskCanvas.dataset.lastY = String(p.y);
  setDirty(true);
});

window.addEventListener("mouseup", () => { drawing = false; });

brushSizeEl.addEventListener("input", () => { updateCursor(); });

zoomRangeEl.addEventListener("input", () => {
  zoom = parseInt(zoomRangeEl.value, 10) / 100.0;
  applyZoom();
  updateCursor();
});

function applyZoom() {
  const w = baseCanvas.width;
  const h = baseCanvas.height;
  const dispW = Math.round(w * zoom);
  const dispH = Math.round(h * zoom);

  baseCanvas.style.width = dispW + "px";
  baseCanvas.style.height = dispH + "px";
  maskCanvas.style.width = dispW + "px";
  maskCanvas.style.height = dispH + "px";
  canvasStackEl.style.width = dispW + "px";
  canvasStackEl.style.height = dispH + "px";
}

function fitZoomToViewerHeight() {
  const imgW = baseCanvas.width;
  const imgH = baseCanvas.height;
  if (!imgW || !imgH) return;

  const availableH = Math.max(100, viewerWrapEl.clientHeight - 32);
  const fit = availableH / imgH;
  zoom = Math.min(4.0, Math.max(0.1, fit));

  zoomRangeEl.value = String(Math.round(zoom * 100));
  applyZoom();
  updateCursor();
}

function cacheBust(url) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}_=${Date.now()}`;
}

function clearViewer() {
  baseCtx.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

  baseCanvas.width = 0;
  baseCanvas.height = 0;
  maskCanvas.width = 0;
  maskCanvas.height = 0;

  canvasStackEl.style.width = "0px";
  canvasStackEl.style.height = "0px";

  currentStem = null;
  currentIndex = 0;
  undoStack = [];
  samPoints = [];
  setDirty(false);
  currentInfoEl.dataset.baseText = "";
  currentInfoEl.textContent = "";
  renderList();
}

async function loadList(opts = {}) {
  const { forceReload = false, autoFit = true } = opts;

  const res = await fetch(cacheBust("/api/images"), { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load image list");

  items = data.items || [];
  countInfoEl.textContent = `Total: ${data.count}`;
  renderList();

  if (items.length === 0) {
    clearViewer();
    setStatus("No images found");
    return;
  }

  let nextIndex = 0;

  if (!forceReload && currentStem) {
    const found = items.indexOf(currentStem);
    if (found >= 0) {
      nextIndex = found;
    } else {
      currentStem = null;
      currentIndex = 0;
    }
  }

  if (forceReload || !currentStem) {
    await loadByIndex(nextIndex, {
      autoFit,
      autoSaveBefore: false,
      forceReload: true,
    });
  } else {
    renderList();
  }
}

function renderList() {
  fileListEl.innerHTML = "";
  items.forEach((stem, idx) => {
    const div = document.createElement("div");
    div.className = "file-item" + (idx === currentIndex ? " active" : "");
    div.textContent = stem;
    div.onclick = async () => {
      await loadByIndex(idx, { autoFit: false, autoSaveBefore: true });
    };
    fileListEl.appendChild(div);
  });
}

async function loadByIndex(idx, opts = {}) {
  const { autoFit = false, autoSaveBefore = true, forceReload = false } = opts;
  if (idx < 0 || idx >= items.length || isLoading) return;

  if (autoSaveBefore && currentStem && isDirty) {
    const ok = await saveCurrent(true);
    if (!ok) return;
  }

  isLoading = true;
  currentIndex = idx;
  currentStem = items[idx];
  renderList();
  setStatus(forceReload ? "Reloading..." : "Loading...");

  let imgUrl = null;
  let maskUrl = null;

  try {
    const stem = encodeURIComponent(currentStem);

    const [imgBlob, maskBlob, metaRes] = await Promise.all([
      fetch(cacheBust(`/api/image/${stem}`), { cache: "no-store" }).then(r => {
        if (!r.ok) throw new Error("Failed to fetch image");
        return r.blob();
      }),
      fetch(cacheBust(`/api/mask/${stem}`), { cache: "no-store" }).then(r => {
        if (!r.ok) throw new Error("Failed to fetch mask");
        return r.blob();
      }),
      fetch(cacheBust(`/api/meta/${stem}`), { cache: "no-store" }).then(r => {
        if (!r.ok) throw new Error("Failed to fetch metadata");
        return r.json();
      }),
    ]);

    imgUrl = URL.createObjectURL(imgBlob);
    maskUrl = URL.createObjectURL(maskBlob);

    const img = await loadImage(imgUrl);
    const mask = await loadImage(maskUrl);

    baseCanvas.width = img.width;
    baseCanvas.height = img.height;
    maskCanvas.width = img.width;
    maskCanvas.height = img.height;

    baseCtx.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

    baseCtx.drawImage(img, 0, 0);
    maskCtx.drawImage(mask, 0, 0);

    samPoints = [];
    undoStack = [];

    const infoText = `${metaRes.stem} | ${metaRes.width}×${metaRes.height} | class ${metaRes.class_id}`;
    currentInfoEl.dataset.baseText = infoText;
    currentInfoEl.textContent = infoText;
    setDirty(false);

    if (autoFit) fitZoomToViewerHeight();
    else applyZoom();

    setStatus(`Loaded: ${currentStem}`);
  } catch (err) {
    console.error(err);
    clearViewer();
    setStatus("Load failed");
    alert(String(err));
  } finally {
    if (imgUrl) URL.revokeObjectURL(imgUrl);
    if (maskUrl) URL.revokeObjectURL(maskUrl);
    isLoading = false;
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Image decode failed"));
    img.src = url;
  });
}

function applyBinaryMaskToMaskCanvas(binaryCanvasOrImage) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = maskCanvas.width;
  tempCanvas.height = maskCanvas.height;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
  tempCtx.drawImage(binaryCanvasOrImage, 0, 0, tempCanvas.width, tempCanvas.height);

  const src = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
  const out = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);

  for (let i = 0; i < src.data.length; i += 4) {
    const v = src.data[i];
    const on = v > 127;
    if (on) {
      out.data[i] = 0;
      out.data[i + 1] = 255;
      out.data[i + 2] = 0;
      out.data[i + 3] = 110;
    } else {
      out.data[i] = 0;
      out.data[i + 1] = 0;
      out.data[i + 2] = 0;
      out.data[i + 3] = 0;
    }
  }

  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskCtx.putImageData(out, 0, 0);
}

async function runSam(silent = false) {
  if (!currentStem || samPoints.length === 0 || isSamRunning) return false;

  isSamRunning = true;
  setStatus("Running SAM...");

  try {
    const res = await fetch(`/api/sam_predict/${encodeURIComponent(currentStem)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: samPoints })
    });

    if (!res.ok) throw new Error(await res.text());

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const samMaskImg = await loadImage(url);

    pushUndo();
    applyBinaryMaskToMaskCanvas(samMaskImg);
    setDirty(true);
    setStatus("SAM applied");

    URL.revokeObjectURL(url);
    return true;
  } catch (err) {
    console.error(err);
    setStatus("SAM failed");
    if (!silent) alert(String(err));
    return false;
  } finally {
    isSamRunning = false;
  }
}

async function saveCurrent(silent = false) {
  if (!currentStem) return false;

  setStatus("Saving...");
  try {
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = maskCanvas.width;
    exportCanvas.height = maskCanvas.height;
    const exportCtx = exportCanvas.getContext("2d");

    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const out = exportCtx.createImageData(maskCanvas.width, maskCanvas.height);

    for (let i = 0; i < maskData.data.length; i += 4) {
      const alpha = maskData.data[i + 3];
      const v = alpha > MASK_DRAW_ALPHA_THRESHOLD ? 255 : 0;
      out.data[i] = v;
      out.data[i + 1] = v;
      out.data[i + 2] = v;
      out.data[i + 3] = 255;
    }

    exportCtx.putImageData(out, 0, 0);

    const blob = await new Promise(resolve => exportCanvas.toBlob(resolve, "image/png"));
    const buf = await blob.arrayBuffer();
    const bytes = Array.from(new Uint8Array(buf));

    const res = await fetch(`/api/save/${encodeURIComponent(currentStem)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mask: bytes })
    });

    if (!res.ok) {
      const t = await res.text();
      setStatus("Save failed");
      if (!silent) alert(t);
      return false;
    }

    setDirty(false);
    setStatus(`Saved: ${currentStem}`);
    return true;
  } catch (err) {
    console.error(err);
    setStatus("Save failed");
    if (!silent) alert(String(err));
    return false;
  }
}

async function prevImage() {
  if (currentIndex > 0) {
    await loadByIndex(currentIndex - 1, { autoFit: false, autoSaveBefore: true });
  }
}

async function nextImage() {
  if (currentIndex < items.length - 1) {
    await loadByIndex(currentIndex + 1, { autoFit: false, autoSaveBefore: true });
  }
}

window.addEventListener("keydown", async (e) => {
  if (e.ctrlKey && e.key.toLowerCase() === "s") {
    e.preventDefault();
    await saveCurrent();
  } else if (e.ctrlKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undoMask();
  } else if (e.key === "1") setMode("brush");
  else if (e.key === "2") setMode("erase");
  else if (e.key === "3") setMode("sam_point");
  else if (e.key.toLowerCase() === "a") await prevImage();
  else if (e.key.toLowerCase() === "d") await nextImage();
});

window.addEventListener("resize", () => {
  if (baseCanvas.width > 0 && baseCanvas.height > 0 && activePipelineTab === "gui") {
    fitZoomToViewerHeight();
  }
});

function setupTrainingOptionProxies() {
  const resumePath = document.getElementById("trainResumePath");
  if (resumePath) {
    const label = resumePath.closest(".field")?.querySelector("label");
    if (label) label.textContent = "Resume checkpoint";
    resumePath.placeholder = "체크포인트 경로 또는 비움";
    resumePath.closest(".field-row")?.classList.add("resume-option-row");
  }

  const pretrainProxy = document.getElementById("trainPretrainedBackboneOnlyProxy");
  const ampSelect = document.getElementById("trainAmp");
  const hiddenResume = document.getElementById("trainResumeBool");
  if (pretrainProxy && ampSelect) {
    const pretrainField = pretrainProxy.closest(".field");
    const runtimeRow = ampSelect.closest(".field-row");
    const hiddenResumeField = hiddenResume?.closest(".field");
    if (pretrainField && runtimeRow) {
      runtimeRow.insertBefore(pretrainField, hiddenResumeField || null);
    }
  }

  const pairs = [
    ["trainResumeAutoProxy", "trainResumeBool"],
    ["trainPretrainedBackboneOnlyProxy", "trainPretrainedBackboneOnly"],
  ];
  pairs.forEach(([proxyId, realId]) => {
    const proxy = document.getElementById(proxyId);
    const real = document.getElementById(realId);
    if (!proxy || !real) return;
    proxy.checked = real.checked;
    proxy.addEventListener("change", () => {
      real.checked = proxy.checked;
    });
  });
}

async function initApp() {
  try {
    setupTrainingOptionProxies();
    await loadPipelineConfig();
    await loadWeightsList(); // 학습용 가중치 모델 목록 로드
    await loadSegmentWeights(); // 세그먼트용 .pt 목록 로드
    switchExtraTab('train'); 
  } catch (err) {
    setPipelineLog(String(err));
  }
  
  const dropZone = document.getElementById("dropZone");
  const filePicker = document.getElementById("filePicker");
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "#4a90e2";
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.style.borderColor = "#9db2c8";
  });
  dropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "#9db2c8";
    await uploadInputFiles(e.dataTransfer.files);
  });
  filePicker.addEventListener("change", async (e) => {
    await uploadInputFiles(e.target.files);
    e.target.value = "";
  });

  switchPipelineTab("input");
  await checkInputStatus();
}

initApp();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
