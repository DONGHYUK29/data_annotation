import sys
import cv2
import numpy as np
from pathlib import Path
import torch

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QScrollArea
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut,
    QPainter, QPen, QColor
)
from PySide6.QtCore import Qt, QTimer, QRect

from segment_anything import sam_model_registry, SamPredictor


# ======================
# Paths
# ======================
BASE_DIR = Path(__file__).resolve().parents[1]  # .../data_annotation

IMAGE_DIR = BASE_DIR / "images" / "original"
MASK_DIR  = BASE_DIR / "images" / "yolo_seg_results" / "masks"
LABEL_DIR = BASE_DIR / "images" / "yolo_seg_results" / "labels"

AFTER_MASK_DIR  = BASE_DIR / "images" / "edited_results" / "masks"
AFTER_LABEL_DIR = BASE_DIR / "images" / "edited_results" / "labels"
AFTER_MASK_DIR.mkdir(parents=True, exist_ok=True)
AFTER_LABEL_DIR.mkdir(parents=True, exist_ok=True)

SAM_WEIGHT = BASE_DIR / "weights" / "sam_vit_b_01ec64.pth"


# ======================
# Label utils
# ======================
HEADER = "# instance_id class_id x1 y1 x2 y2 poly_x0 poly_y0 poly_x1 poly_y1 ..."

def read_label_lines(path: Path):
    if not path.exists():
        return [HEADER]
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [HEADER]
    if not lines[0].startswith("#"):
        lines.insert(0, HEADER)
    return lines

def upsert_instance_line(lines: list[str], instance_id: str, new_line: str):
    out = []
    replaced = False
    for ln in lines:
        if ln.startswith("#"):
            out.append(ln)
            continue
        parts = ln.split()
        if parts and parts[0] == instance_id:
            out.append(new_line)
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(new_line)
    return out

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

    eps = 1.0
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        poly = cnt.reshape(-1, 2).astype(np.float32)
    else:
        poly = approx

    polygon = [(float(x), float(y)) for x, y in poly]
    return (x1, y1, x2, y2), polygon

def parse_instance_id_from_stem(stem: str) -> str:
    """
    Handles:
      - 2_obj0_mask
      - 2_obj0_mask_edit
      - 2_obj0_mask_edited
    Returns: 2_obj0
    """
    name = stem
    for suf in ("_edited", "_edit"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if name.endswith("_mask"):
        name = name[: -len("_mask")]
    return name

def find_src_label_path(image_id: str) -> Path:
    """
    Prefer *_detected.txt, fallback to *.txt
    """
    p = LABEL_DIR / f"{image_id}_detected.txt"
    if p.exists():
        return p
    p2 = LABEL_DIR / f"{image_id}.txt"
    return p2

def get_class_id_from_src_lines(src_lines: list[str]) -> int | None:
    """
    Your current label format uses only '2' (image_id) as first token,
    so we simply take the first non-header line's class_id.
    If multiple lines exist, this picks the first one's class_id.
    """
    for ln in src_lines:
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            try:
                return int(float(parts[1]))
            except ValueError:
                return None
    return None


# ======================
# SAM
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHT))
sam.to(device)
predictor = SamPredictor(sam)


# ======================
# Unified Canvas
# ======================
class UnifiedCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)

        self.image_rgb = None
        self.image_bgr = None
        self.mask = None

        self.image_qimg = None
        self.image_pix = None

        self.overlay_np = None
        self.overlay_qimg = None

        self.scale_factor = 1.0
        self.brush_size = 10
        self.mode = "brush"
        self.drawing = False
        self.last_xy = None

        self.sam_points = []
        self.sam_labels = []

        self.mask_history = []
        self.max_undo = 30

        self.cursor_pos = None
        self.overlay_alpha = 110

    def set_data(self, image_bgr, mask_u8):
        self.image_bgr = np.ascontiguousarray(image_bgr)
        self.image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self.mask = np.ascontiguousarray(mask_u8)

        h, w, _ = self.image_rgb.shape
        self.image_qimg = QImage(self.image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.image_pix = QPixmap.fromImage(self.image_qimg)

        self.overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
        self.overlay_np[..., 1] = 255
        self.overlay_qimg = QImage(self.overlay_np.data, w, h, 4 * w, QImage.Format_ARGB32)

        self.rebuild_overlay_full()

        self.scale_factor = 1.0
        self.mask_history = []
        self.last_xy = None

        self.sam_points = []
        self.sam_labels = []
        predictor.set_image(cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB))

        self.update_widget_size()
        self.update()

    def update_widget_size(self):
        if self.image_pix is None:
            return
        w = int(self.image_pix.width() * self.scale_factor)
        h = int(self.image_pix.height() * self.scale_factor)
        self.resize(w, h)

    def get_mask(self):
        return self.mask

    def set_mode(self, mode: str):
        self.mode = mode
        self.drawing = False
        self.last_xy = None

    def set_brush_size(self, value: int):
        self.brush_size = int(value)

    def push_undo(self):
        if self.mask is None:
            return
        self.mask_history.append(self.mask.copy())
        if len(self.mask_history) > self.max_undo:
            self.mask_history.pop(0)

    def undo(self):
        if not self.mask_history:
            return
        self.mask = self.mask_history.pop()
        self.rebuild_overlay_full()
        self.update()

    def rebuild_overlay_full(self):
        if self.mask is None or self.overlay_np is None:
            return
        self.overlay_np[..., 3] = np.where(self.mask > 0, self.overlay_alpha, 0).astype(np.uint8)

    def update_overlay_dirty(self, x1, y1, x2, y2):
        if self.mask is None or self.overlay_np is None:
            return
        h, w = self.mask.shape
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 < x1 or y2 < y1:
            return

        m = self.mask[y1:y2 + 1, x1:x2 + 1]
        self.overlay_np[y1:y2 + 1, x1:x2 + 1, 3] = np.where(m > 0, self.overlay_alpha, 0).astype(np.uint8)

        dirty = QRect(
            int(x1 * self.scale_factor),
            int(y1 * self.scale_factor),
            int((x2 - x1 + 1) * self.scale_factor),
            int((y2 - y1 + 1) * self.scale_factor),
        )
        self.update(dirty)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            self.scale_factor *= 1.1 if event.angleDelta().y() > 0 else 0.9
            self.scale_factor = max(0.3, min(self.scale_factor, 5.0))
            self.update_widget_size()
            self.update()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position().toPoint()
        if self.mode in ("brush", "erase") and self.drawing:
            self.paint_brush(event)
        else:
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.image_rgb is None or self.mask is None:
            return

        if self.mode in ("brush", "erase"):
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.push_undo()
                self.last_xy = None
                self.paint_brush(event)
            return

        if self.mode == "sam":
            x = int(event.position().x() / self.scale_factor)
            y = int(event.position().y() / self.scale_factor)

            h, w = self.mask.shape
            if not (0 <= x < w and 0 <= y < h):
                return

            if event.button() == Qt.LeftButton:
                self.sam_points.append([x, y])
                self.sam_labels.append(1)
            elif event.button() == Qt.RightButton:
                self.sam_points.append([x, y])
                self.sam_labels.append(0)
            else:
                return

            self.push_undo()
            self.run_sam()
            return

    def mouseReleaseEvent(self, event):
        self.drawing = False
        self.last_xy = None

    def paint_brush(self, event):
        x = int(event.position().x() / self.scale_factor)
        y = int(event.position().y() / self.scale_factor)

        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        r = int(self.brush_size)
        color = 255 if self.mode == "brush" else 0

        if self.last_xy is None:
            cv2.circle(self.mask, (x, y), r, int(color), -1)
            self.update_overlay_dirty(x - r, y - r, x + r, y + r)
            self.last_xy = (x, y)
            return

        x0, y0 = self.last_xy
        cv2.line(self.mask, (x0, y0), (x, y), int(color), thickness=max(1, 2 * r))
        cv2.circle(self.mask, (x, y), r, int(color), -1)

        x1 = min(x0, x) - r - 2
        y1 = min(y0, y) - r - 2
        x2 = max(x0, x) + r + 2
        y2 = max(y0, y) + r + 2
        self.update_overlay_dirty(x1, y1, x2, y2)

        self.last_xy = (x, y)

    def run_sam(self):
        if self.image_bgr is None or len(self.sam_points) == 0:
            return
        input_points = np.array(self.sam_points)
        input_labels = np.array(self.sam_labels)

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

        self.mask = (masks[0] * 255).astype("uint8")
        self.rebuild_overlay_full()
        self.update()

    def paintEvent(self, event):
        if self.image_pix is None or self.overlay_qimg is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, self.image_pix)
        painter.drawImage(0, 0, self.overlay_qimg)

        if self.cursor_pos is not None:
            painter.resetTransform()
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            if self.mode in ("brush", "erase"):
                r = int(self.brush_size * self.scale_factor)
            else:
                r = 8
            painter.drawEllipse(self.cursor_pos, r, r)


# ======================
# Main Window
# ======================
class UnifiedEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Editor - Unified (Brush/Eraser/SAM)")
        self.resize(1900, 1000)

        self.mask_files = sorted(MASK_DIR.glob("*.png"))
        if not self.mask_files:
            print("No mask files found in:", MASK_DIR)
            sys.exit(0)

        self.index = 0
        self.current_name = None

        self.canvas = UnifiedCanvas()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)

        btn_brush = QPushButton("Brush")
        btn_erase = QPushButton("Eraser")
        btn_sam   = QPushButton("SAM")
        btn_save   = QPushButton("Save")
        btn_prev  = QPushButton("Prev")
        btn_next  = QPushButton("Next")

        btn_brush.clicked.connect(lambda: self.canvas.set_mode("brush"))
        btn_erase.clicked.connect(lambda: self.canvas.set_mode("erase"))
        btn_sam.clicked.connect(lambda: self.canvas.set_mode("sam"))
        btn_save.clicked.connect(self.auto_save)
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(80)
        slider.setValue(10)
        slider.valueChanged.connect(self.canvas.set_brush_size)

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout()
        root.setLayout(layout)
        layout.addWidget(self.scroll)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_brush)
        btn_layout.addWidget(btn_erase)
        btn_layout.addWidget(btn_sam)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)
        layout.addLayout(btn_layout)
        layout.addWidget(slider)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        self.load_current()

    def auto_save(self):
        mask = self.canvas.get_mask()
        if mask is None or self.current_name is None:
            return

        stem = self.current_name  # e.g. "2_obj0_mask"
        instance_id = parse_instance_id_from_stem(stem)  # "2_obj0"
        image_id = instance_id.split("_obj")[0]          # "2"

        # (1) Save edited mask
        save_mask_path = AFTER_MASK_DIR / f"{stem}_edited.png"
        cv2.imwrite(str(save_mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # (2) Load source labels (class_id is keyed by image_id-only in your current format)
        src_label_path = find_src_label_path(image_id)
        src_lines = read_label_lines(src_label_path)
        class_id = get_class_id_from_src_lines(src_lines)

        if class_id is None:
            print(f"[WARN] class_id not found in {src_label_path} (your label first token is image_id-only)")
            return

        # (3) Recompute bbox and polygon
        bbox, poly = mask_to_bbox_and_polygon(mask)
        if bbox is None or poly is None or len(poly) < 3:
            print(f"[WARN] empty/invalid mask after edit: {instance_id}")
            return

        x1, y1, x2, y2 = bbox

        poly_flat = []
        for (px, py) in poly:
            poly_flat.append(f"{px:.2f}")
            poly_flat.append(f"{py:.2f}")

        # IMPORTANT: we write edited label with instance_id = "2_obj0" (consistent with mask naming)
        new_line = (
            f"{instance_id} {class_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            + " ".join(poly_flat)
        )

        # (4) Upsert into edited label file
        dst_label_path = AFTER_LABEL_DIR / f"{image_id}_edited.txt"

        if dst_label_path.exists():
            dst_lines = read_label_lines(dst_label_path)
        else:
            # start from source labels BUT convert any "image_id-only" rows into instance rows is not possible.
            # So we start a fresh file with header.
            dst_lines = [HEADER]

        dst_lines = upsert_instance_line(dst_lines, instance_id, new_line)
        dst_label_path.write_text("\n".join(dst_lines) + "\n", encoding="utf-8")

        print(f"[SAVE] {save_mask_path.name} + {dst_label_path.name}")

    def focus_on_mask(self):
        mask = self.canvas.get_mask()
        if mask is None:
            return

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return

        cx = int(((xs.min() + xs.max()) // 2) * self.canvas.scale_factor)
        cy = int(((ys.min() + ys.max()) // 2) * self.canvas.scale_factor)

        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()

        h_bar.setValue(max(0, cx - self.scroll.viewport().width() // 2))
        v_bar.setValue(max(0, cy - self.scroll.viewport().height() // 2))

    def load_current(self):
        mask_path = self.mask_files[self.index]
        stem = mask_path.stem
        image_key = stem.split("_obj")[0]

        image_path = None
        for ext in ["jpg", "png", "jpeg"]:
            cand = IMAGE_DIR / f"{image_key}.{ext}"
            if cand.exists():
                image_path = cand
                break
        if image_path is None:
            print("Image not found for:", stem)
            return

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print("Failed to read image:", image_path)
            return

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Failed to read mask:", mask_path)
            return
        mask = (mask > 127).astype(np.uint8) * 255

        self.current_name = stem
        self.canvas.set_data(img_bgr, mask)

        QTimer.singleShot(0, self.focus_on_mask)

    def next_image(self):
        self.auto_save()
        if self.index < len(self.mask_files) - 1:
            self.index += 1
            self.load_current()
        else:
            self.close()

    def prev_image(self):
        self.auto_save()
        if self.index > 0:
            self.index -= 1
            self.load_current()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UnifiedEditor()
    win.showMaximized()
    sys.exit(app.exec())