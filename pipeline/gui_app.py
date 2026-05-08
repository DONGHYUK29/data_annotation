from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import threading
import uuid
import mimetypes
import time
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from PIL import Image

import config as cfg
from .sam_backend import predict_sam_mask

cfg.ensure_stage_dirs()



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


IMAGE_DIR = cfg.IMAGES_RAW_DIR
MASK_DIR = cfg.STAGE1_MASKS
LABEL_DIR = cfg.STAGE1_LABELS

AFTER_IMAGE_DIR = cfg.STAGE2_IMAGES
AFTER_MASK_DIR = cfg.STAGE2_MASKS
AFTER_LABEL_DIR = cfg.STAGE2_LABELS

app = FastAPI(title="Annotation Web GUI + SAM Point Assist", lifespan=lifespan)

MASK_ALPHA = 110
MASK_RGBA = (0, 255, 0, MASK_ALPHA)

PIPELINE_STEPS = ("extract", "segment", "gui", "export")

PIPELINE_DIRS = {
    "extract_rgbd": cfg.INPUT_RGBD_DIR,
    "extract_images": cfg.INPUT_IMAGES_DIR,
    
    "segment_images": cfg.STAGE1_DIR / "images",
    "segment_labels": cfg.STAGE1_LABELS,
    "segment_masks": cfg.STAGE1_MASKS,
    
    "export_rgbd": cfg.DATASET_DIR / "rgbd",
    "export_images": cfg.DATASET_DIR / "images",
    "export_labels": cfg.DATASET_DIR / "labels",
    "export_masks": cfg.DATASET_DIR / "masks",
}

JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}


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
        JOBS[job_id]["status"] = "running"
    try:
        proc = subprocess.run(
            " ".join(cmd),
            cwd=str(cfg.PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            shell=True,
        )
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
            JOBS[job_id]["returncode"] = proc.returncode
            JOBS[job_id]["output"] = output.strip()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["returncode"] = -1
            JOBS[job_id]["output"] = str(exc)


def _build_docker_command(step: str, payload: dict) -> list[str]:
    base = ["docker", "compose", "run", "--rm"]
    if step == "gui":
        base.append("--service-ports")
    cmd = ["python", "run.py", step]

    if step == "extract":
        cmd.extend(["--start", str(int(payload.get("start", 0)))])
        cmd.extend(["--end", str(int(payload.get("end", 9)))])
        cmd.extend(["--count", str(int(payload.get("count", 100)))])
    elif step == "segment":
        if payload.get("input_dir"):
            cmd.extend(["--input", str(payload["input_dir"])])
        if payload.get("output_dir"):
            cmd.extend(["--output", str(payload["output_dir"])])
    elif step == "gui":
        cmd.extend(["--host", "0.0.0.0", "--port", str(cfg.WEB_PORT)])
    elif step == "export":
        cmd.extend(["--bg", str(payload.get("bg_prefix", "Environment"))])
        cmd.extend(["--mode", str(payload.get("export_mode", "copy"))])
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


def png_response_from_array(arr: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encoding failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/check_bags")
def api_check_bags():
    bag_dir = cfg.BAG_DIR
    if not bag_dir.exists():
        return {"count": 0, "files": []}
    bags = list(bag_dir.glob("*.bag"))
    bags.sort(key=lambda x: x.name)
    return {
        "count": len(bags),
        "files": [b.name for b in bags]
    }


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


@app.get("/api/images")
def api_images():
    files = list_input_images()
    return {
        "items": [p.stem for p in files],
        "count": len(files),
    }


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
        "class_id": stem_to_class_id(stem),
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
    class_id = stem_to_class_id(stem)

    if bbox is not None and poly is not None and len(poly) >= 3:
        poly_norm: list[str] = []
        for px, py in poly:
            xn = px / max(w - 1, 1)
            yn = py / max(h - 1, 1)
            poly_norm.append(f"{xn:.6f}")
            poly_norm.append(f"{yn:.6f}")

        x1, y1, x2, y2 = bbox
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        bbox_norm = [f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]

        line = str(class_id) + " " + " ".join(bbox_norm + poly_norm)

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
            "extract_start": 0,
            "extract_end": 9,
            "extract_count": 100,
            "export_bg_prefix": "Environment",
            "export_mode": "copy",
        },
    }

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
    
    dirs.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())

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

# Web UI
HTML_PAGE = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Annotation Web GUI + SAM Point Assist</title>
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

    /* Right Sidebar Tabs */
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

    /* Right Sidebar Settings Block */
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

    /* Center containers */
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

    /* 부모 컨테이너 (Flex 유지) */
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

    /* 이미지 미리보기 */
    .preview-content img {
      width: 100%;
      height: 100%;
      object-fit: contain; 
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* 텍스트 파일 미리보기 */
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

    #pipelineLog {
      flex: 1;
      width: 100%;
      min-height: 180px;
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
      grid-template-columns: 1fr 1fr;
      border-top: 1px solid #d9dee5;
      background: #fff;
      min-height: 64px;
    }
    .bottom-left, .bottom-right {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      box-sizing: border-box;
      min-width: 0;
      flex-wrap: wrap;
    }
    .bottom-left { border-right: 1px solid #e5e7eb; }
    .bottom-right { justify-content: space-between; }

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

    /* Left Sidebar Folder Views */
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
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar" id="leftSidebar">
    
    <div id="homeSidebarContent" style="display: none;">
      <h3 style="margin-top: 0;">Welcome</h3>
      <p class="muted">우측 패널에서 파이프라인 단계를 선택하세요.</p>
    </div>

    <div id="guiSidebarContent" style="display: none;">
      <h3 style="margin-top: 0;">Images</h3>
      <div class="muted" id="countInfo"></div>

      <div class="hint-box">
        <b>Copy Unedited to Output2</b><br>
        아직 수정하지 않은 이미지들에 대해<br>
        output_1 결과를 output_2로 그대로 복사합니다.<br>
        즉, 편집 안 한 것도 최종본으로 넘기는 기능입니다.
      </div>
      <div style="margin: 10px 0;">
        <button id="btnMoveRemaining" onclick="moveRemaining(event)" style="width:100%">Copy Unedited to Output2</button>
      </div>

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
    
    <div id="homeContainer" class="center-tab-content">
      <h2 style="margin-top:0">Pipeline Initialization</h2>
      <div id="bagCheckResult" class="check-result-box">
        <span class="muted">Checking bag files...</span>
      </div>
    </div>

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
          <button id="brushBtn" class="active" onclick="setMode('brush')">🖌️ Brush</button>
          <button id="eraseBtn" onclick="setMode('erase')">🧽 Eraser</button>
          <button id="samPointBtn" onclick="setMode('sam_point')">🎯 SAM Point</button>
          <label>Brush Size <input type="range" id="brushSize" min="1" max="80" value="10" /></label>
          <button onclick="undoMask()">Undo</button>
          <button onclick="clearMask()">Erase All</button>
          <label>Zoom <input type="range" id="zoomRange" min="10" max="400" value="100" /></label>
          <span id="status" class="muted">Ready</span>
        </div>
        <div class="bottom-right">
          <div>
            <button onclick="prevImage()">◀ Prev</button>
            <button onclick="nextImage()">Next ▶</button>
            <button onclick="saveCurrent()">💾 Save</button>
          </div>
          <div><span id="currentInfo" class="muted"></span></div>
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
    <h3 style="margin-top: 0;">Pipeline Controls</h3>
    
    <div class="right-tabs">
      <button onclick="switchPipelineTab('extract')" id="tabBtn_extract">Extract</button>
      <button onclick="switchPipelineTab('segment')" id="tabBtn_segment">Segment</button>
      <button onclick="switchPipelineTab('gui')" id="tabBtn_gui">GUI</button>
      <button onclick="switchPipelineTab('export')" id="tabBtn_export">Export</button>
    </div>
    
    <div id="rightSettingsArea">
      
      <div id="extractSettings" class="settings-block" style="display: none;">
        <h4>Extract Settings</h4>
        <div class="field"><label>Start Bag Index</label><input id="extractStart" type="number" min="0" value="0" /></div>
        <div class="field"><label>End Bag Index</label><input id="extractEnd" type="number" min="0" value="9" /></div>
        <div class="field"><label>Frames Per Bag</label><input id="extractCount" type="number" min="1" value="100" /></div>
        <button class="run-btn" onclick="runPipelineStep('extract')">▶ Run Extract</button>
      </div>
      
      <div id="segmentSettings" class="settings-block" style="display: none;">
        <h4>Segment Settings</h4>
        <div class="muted" style="margin-bottom: 12px;">Uses default input/output paths.</div>
        <button class="run-btn" onclick="runPipelineStep('segment')">▶ Run Segment</button>
      </div>

      <div id="exportSettings" class="settings-block" style="display: none;">
        <h4>Export Settings</h4>
        <div class="field"><label>Export Prefix (bg)</label><input id="exportBgPrefix" type="text" value="Environment" /></div>
        <div class="field">
          <label>Mode</label>
          <select id="exportMode">
            <option value="copy">copy</option>
            <option value="move">move</option>
          </select>
        </div>
        <button class="run-btn" onclick="runPipelineStep('export')">▶ Run Export</button>
      </div>
      
    </div>
    
    <h4 style="margin-top: 24px; margin-bottom: 8px;">Log</h4>
    <div id="pipelineLog">Waiting...</div>
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

const stepFolderMap = {
    extract: [
        { label: "create/input/rgbd/", key: "extract_rgbd" },
        { label: "create/input/images/", key: "extract_images" }
    ],
    segment: [
        { label: "create/output1/images/", key: "segment_images" },
        { label: "create/output1/labels/", key: "segment_labels" },
        { label: "create/output1/masks/", key: "segment_masks" }
    ],
    export: [
        { label: "create/dataset/rgbd", key: "export_rgbd" },
        { label: "create/dataset/images", key: "export_images" },
        { label: "create/dataset/labels", key: "export_labels" },
        { label: "create/dataset/masks", key: "export_masks" }
    ]
};

function setStatus(msg) { statusEl.textContent = msg; }
function setPipelineLog(msg) { pipelineLogEl.textContent = msg || "No logs"; }

// -- Bag 파일 체크 API 호출 --
async function checkBags() {
    try {
        const res = await fetch('/api/check_bags');
        const data = await res.json();
        const container = document.getElementById("bagCheckResult");
        
        if (data.count > 0) {
            container.innerHTML = `
                <h3 style="color: #2e7d32; margin-top:0;">✅ ${data.count}개의 bag 파일을 찾았습니다.</h3>
                <ul>${data.files.map(f => `<li>${f}</li>`).join('')}</ul>
                <p style="margin-top: 16px; color: #475467;">우측의 <b>Extract</b> 탭을 눌러 파이프라인을 시작하세요.</p>
            `;
        } else {
            container.innerHTML = `
                <h3 style="color: #d32f2f; margin-top:0;">⚠️ bag 파일을 찾을 수 없습니다.</h3>
                <p style="color: #475467; margin-bottom: 0;">
                    <code>bag/</code> 경로에 <code>class_number.bag</code> 형식의 RealSense raw bag 파일을 추가해주세요.
                </p>
            `;
        }
    } catch (e) {
        console.error(e);
        document.getElementById("bagCheckResult").innerHTML = '<span style="color:red;">Failed to check bag files.</span>';
    }
}

// -- 중앙 화면 미리보기 API 호출 --
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

// -- 파일/탐색기 트리 토글 함수 --
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

// -- 왼쪽 사이드바 베이스 폴더 렌더링 --
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

// Copy Unedited to Output 2 실행 함수
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
    setStatus(`Copied ${data.moved} items`);
    alert(`성공적으로 처리되었습니다.\nCopied unedited items to output_2: ${data.moved}`);
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

// -- 탭 스위칭 (확인창 및 Copy Unedited 기능 포함) --
function switchPipelineTab(step) {
    // GUI에서 다른 단계로 이동 시 알림 및 Confirm -> Copy 실행
    if (activePipelineTab === 'gui' && step !== 'gui' && step !== '') {
        const msg = "GUI 편집을 마치고 다른 단계로 넘어가기 전에,\n아직 편집하지 않은(Unedited) 마스크들을 Output 2(최종본)로 일괄 복사하시겠습니까?\n\n'확인'을 누르시면 'Copy Unedited to Output 2'가 실행됩니다.";
        
        if (confirm(msg)) {
            moveRemaining();
        }
    }

    activePipelineTab = step;
    const tabs = ["extract", "segment", "gui", "export"];
  
    // 1. Right Sidebar Tab Buttons & Settings Block Toggles
    tabs.forEach(t => {
        const btn = document.getElementById("tabBtn_" + t);
        if (btn) btn.classList.toggle("active", t === step);
        
        const settingsBlock = document.getElementById(t + "Settings");
        if (settingsBlock) {
            settingsBlock.style.display = (t === step) ? "block" : "none";
        }
    });

    // 2. Center Container Toggles
    document.getElementById("homeContainer").style.display = (step === "") ? "block" : "none";
    document.getElementById("guiContainer").style.display = (step === "gui") ? "grid" : "none";
    document.getElementById("previewContainer").style.display = (step !== "" && step !== "gui") ? "flex" : "none";

    // 3. Left Sidebar Toggles
    document.getElementById("homeSidebarContent").style.display = (step === "") ? "block" : "none";
    document.getElementById("guiSidebarContent").style.display = (step === "gui") ? "block" : "none";
    document.getElementById("folderSidebarContent").style.display = (step !== "" && step !== "gui") ? "block" : "none";

    // 4. Data Reload based on Context
    if (step === "") {
        checkBags();
    } else if (step === "gui") {
        loadList(); 
        if (baseCanvas.width > 0) fitZoomToViewerHeight();
    } else {
        document.getElementById("previewContent").innerHTML = '<span class="muted">Select a file from the left sidebar to preview</span>';
        document.getElementById("folderSidebarTitle").textContent = step.toUpperCase() + " Paths";
        renderLeftFolders(step); 
    }
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/api/pipeline/job/${encodeURIComponent(jobId)}`);
    const data = await res.json();
    if (!res.ok) {
      setPipelineLog(data.detail || "Job poll failed");
      return;
    }
    const command = Array.isArray(data.command) ? data.command.join(" ") : "";
    const output = data.output || "";
    setPipelineLog(`[${data.status}] ${command}\n\n${output}`);
    if (data.status === "done" || data.status === "failed") {
      setStatus(`Pipeline ${data.step}: ${data.status}`);
      return;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function runPipelineStep(step) {
  const payload = { step };
  if (step === "extract") {
    payload.start = parseInt(document.getElementById("extractStart").value || "0", 10);
    payload.end = parseInt(document.getElementById("extractEnd").value || "9", 10);
    payload.count = parseInt(document.getElementById("extractCount").value || "100", 10);
  } else if (step === "export") {
    payload.bg_prefix = document.getElementById("exportBgPrefix").value || "Environment";
    payload.export_mode = document.getElementById("exportMode").value || "copy";
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
    await pollJob(currentJobId);
  } catch (err) {
    setStatus(`Failed to start ${step}`);
    alert(String(err));
  }
}

async function loadPipelineConfig() {
  const res = await fetch("/api/pipeline/config");
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load pipeline config");
  pipelineConfig = data;
  
  // Set default values in DOM (서버로부터 받은 defaults 사용)
  document.getElementById("extractStart").value = data.defaults.extract_start;
  document.getElementById("extractEnd").value = data.defaults.extract_end;
  document.getElementById("extractCount").value = data.defaults.extract_count;
  document.getElementById("exportBgPrefix").value = data.defaults.export_bg_prefix;
  document.getElementById("exportMode").value = data.defaults.export_mode;
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

async function loadList() {
  const res = await fetch("/api/images");
  const data = await res.json();
  items = data.items || [];
  countInfoEl.textContent = `Total: ${data.count}`;
  renderList();

  if (items.length > 0 && !currentStem) {
    await loadByIndex(0, { autoFit: true, autoSaveBefore: false });
  } else if (items.length === 0) {
    setStatus("No images found");
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
  const { autoFit = false, autoSaveBefore = true } = opts;
  if (idx < 0 || idx >= items.length || isLoading) return;

  if (autoSaveBefore && currentStem && isDirty) {
    const ok = await saveCurrent(true);
    if (!ok) return;
  }

  isLoading = true;
  currentIndex = idx;
  currentStem = items[idx];
  renderList();
  setStatus("Loading...");

  try {
    const [imgBlob, maskBlob, metaRes] = await Promise.all([
      fetch(`/api/image/${encodeURIComponent(currentStem)}`).then(r => {
        if (!r.ok) throw new Error("Failed to fetch image");
        return r.blob();
      }),
      fetch(`/api/mask/${encodeURIComponent(currentStem)}`).then(r => {
        if (!r.ok) throw new Error("Failed to fetch mask");
        return r.blob();
      }),
      fetch(`/api/meta/${encodeURIComponent(currentStem)}`).then(r => {
        if (!r.ok) throw new Error("Failed to fetch metadata");
        return r.json();
      }),
    ]);

    const imgUrl = URL.createObjectURL(imgBlob);
    const maskUrl = URL.createObjectURL(maskBlob);

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

    URL.revokeObjectURL(imgUrl);
    URL.revokeObjectURL(maskUrl);

    setStatus(`Loaded: ${currentStem}`);
  } catch (err) {
    console.error(err);
    setStatus("Load failed");
    alert(String(err));
  } finally {
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

async function initApp() {
  try {
    await loadPipelineConfig();
  } catch (err) {
    setPipelineLog(String(err));
  }
  
  // 초기화면 렌더링 (빈 탭)
  switchPipelineTab("");
}

initApp();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE