import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


ROOT_DIR = Path(r"C:\Users\user\Desktop\pj\research\data_annotation_sw\create\dataset\rgbd")


def normalize_depth_for_display(depth):
    depth = depth.astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.uint8)

    valid_depth = depth[valid]

    d_min = np.percentile(valid_depth, 2)
    d_max = np.percentile(valid_depth, 98)

    if d_max <= d_min:
        return np.zeros_like(depth, dtype=np.uint8)

    clipped = np.clip(depth, d_min, d_max)
    norm = (clipped - d_min) / (d_max - d_min)
    norm = (norm * 255).clip(0, 255).astype(np.uint8)
    return norm


def find_rgb_depth_keys(data):
    keys = list(data.keys())
    print(f"NPZ keys: {keys}")

    rgb_key = None
    depth_key = None

    # 1차: 이름 기반
    for k in keys:
        lk = k.lower()
        if rgb_key is None and ("rgb" in lk or "bgr" in lk or "color" in lk or "image" in lk):
            rgb_key = k
        if depth_key is None and "depth" in lk:
            depth_key = k

    # 2차: shape 기반
    for k in keys:
        arr = data[k]
        if rgb_key is None and arr.ndim == 3 and arr.shape[-1] in (3, 4):
            rgb_key = k
        if depth_key is None and arr.ndim == 2:
            depth_key = k

    return rgb_key, depth_key


def prepare_bgr_and_rgb(img):
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # 네가 BGR-D라고 했으니 원본은 BGR로 간주
    bgr = img
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def resize_depth_to_match(depth, target_hw):
    th, tw = target_hw
    if depth.shape[:2] == (th, tw):
        return depth
    return cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)


def make_depth_overlay_on_bgr(bgr, depth_vis, alpha=0.45):
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # BGR
    overlay = cv2.addWeighted(bgr, 1.0 - alpha, depth_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def make_depth_edge_overlay_on_bgr(bgr, depth_vis):
    edges = cv2.Canny(depth_vis, 50, 150)

    edge_bgr = bgr.copy()
    edge_bgr[edges > 0] = (0, 0, 255)  # 빨간 에지 (BGR)

    edge_rgb = cv2.cvtColor(edge_bgr, cv2.COLOR_BGR2RGB)
    return edge_rgb


def visualize_one_npz(npz_path):
    print(f"\n[OPEN] {npz_path}")
    data = np.load(npz_path)

    rgb_key, depth_key = find_rgb_depth_keys(data)

    if rgb_key is None:
        raise ValueError("RGB/BGR 데이터를 찾지 못했습니다.")
    if depth_key is None:
        raise ValueError("Depth 데이터를 찾지 못했습니다.")

    color_raw = data[rgb_key]
    depth = data[depth_key]

    print(f"Color key : {rgb_key}, shape={color_raw.shape}, dtype={color_raw.dtype}")
    print(f"Depth key : {depth_key}, shape={depth.shape}, dtype={depth.dtype}")

    bgr, rgb = prepare_bgr_and_rgb(color_raw)
    depth = resize_depth_to_match(depth, bgr.shape[:2])
    depth_vis = normalize_depth_for_display(depth)

    overlay_rgb = make_depth_overlay_on_bgr(bgr, depth_vis, alpha=0.45)
    edge_overlay_rgb = make_depth_edge_overlay_on_bgr(bgr, depth_vis)

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(rgb)
    plt.title(f"Color ({rgb_key})")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(depth_vis, cmap="jet")
    plt.title(f"Depth ({depth_key})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(overlay_rgb)
    plt.title("BGR-D Overlay")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(edge_overlay_rgb)
    plt.title("Depth Edge on BGR")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {ROOT_DIR}")

    npz_files = sorted(ROOT_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f".npz 파일이 없습니다: {ROOT_DIR}")

    print(f"총 {len(npz_files)}개 파일 발견")

    for i, npz_path in enumerate(npz_files, 1):
        print(f"\n===== [{i}/{len(npz_files)}] =====")
        visualize_one_npz(npz_path)


if __name__ == "__main__":
    main()