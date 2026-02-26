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
# 경로 설정 (images/* 구조 사용)
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # .../data_annotation
IMAGE_DIR = BASE_DIR / "images" / "original"
MASK_DIR  = BASE_DIR / "images" / "yolo_seg_results" / "masks"
AFTER_DIR = BASE_DIR / "images" / "edit_results" / "masks"
AFTER_DIR.mkdir(parents=True, exist_ok=True)

SAM_WEIGHT = BASE_DIR / "weights/sam_vit_b_01ec64.pth"


# ======================
# SAM 로드
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHT))
sam.to(device)
predictor = SamPredictor(sam)


# ======================
# Unified Canvas
# ======================
class UnifiedCanvas(QWidget):
    """
    mode:
      - "brush" : 좌클릭 드래그로 mask=255
      - "erase" : 좌클릭 드래그로 mask=0
      - "sam"   : 좌클릭=FG(+), 우클릭=BG(-) 점 추가 후 SAM 실행
    """
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)

        # data
        self.image_rgb = None            # np (H,W,3) RGB
        self.image_bgr = None            # np (H,W,3) BGR (SAM set_image용)
        self.mask = None                 # np (H,W) uint8 0/255

        # render cache
        self.image_qimg = None
        self.image_pix = None

        self.overlay_np = None           # (H,W,4) ARGB
        self.overlay_qimg = None

        # interaction
        self.scale_factor = 1.0
        self.brush_size = 10
        self.mode = "brush"
        self.drawing = False
        self.last_xy = None

        # SAM prompts
        self.sam_points = []  # [[x,y],...]
        self.sam_labels = []  # [1/0,...]

        # undo
        self.mask_history = []
        self.max_undo = 30

        # cursor
        self.cursor_pos = None
        self.overlay_alpha = 110

        # 성능: 드래그 중 커서 업데이트 과다 방지(필요시)
        # self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def set_data(self, image_bgr, mask_u8):
        self.image_bgr = np.ascontiguousarray(image_bgr)
        self.image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self.mask = np.ascontiguousarray(mask_u8)

        h, w, _ = self.image_rgb.shape

        self.image_qimg = QImage(self.image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.image_pix = QPixmap.fromImage(self.image_qimg)

        self.overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
        self.overlay_np[..., 1] = 255  # green
        self.overlay_qimg = QImage(self.overlay_np.data, w, h, 4 * w, QImage.Format_ARGB32)

        self.rebuild_overlay_full()

        self.scale_factor = 1.0
        self.mask_history = []
        self.last_xy = None

        # SAM 초기화
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
        # SAM 모드로 전환 시, 기존 점은 유지(원하면 여기서 초기화 가능)
        # if mode == "sam":
        #     self.sam_points, self.sam_labels = [], []

    def set_brush_size(self, value: int):
        self.brush_size = int(value)

    # ---------- Undo ----------
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

    # ---------- Overlay ----------
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

    # ---------- Zoom ----------
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            self.scale_factor *= 1.1 if event.angleDelta().y() > 0 else 0.9
            self.scale_factor = max(0.3, min(self.scale_factor, 5.0))
            self.update_widget_size()
            self.update()
        else:
            event.ignore()

    # ---------- Mouse ----------
    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position().toPoint()

        if self.mode in ("brush", "erase") and self.drawing:
            self.paint_brush(event)
        else:
            # 커서만
            self.update()

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.image_rgb is None or self.mask is None:
            return

        # BRUSH / ERASER
        if self.mode in ("brush", "erase"):
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.push_undo()
                self.last_xy = None
                self.paint_brush(event)
            return

        # SAM
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

            # SAM 실행은 mask를 크게 바꾸므로 undo 스냅샷
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

    # ---------- SAM ----------
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

    # ---------- Paint ----------
    def paintEvent(self, event):
        if self.image_pix is None or self.overlay_qimg is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 이미지 좌표계로 그림
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, self.image_pix)
        painter.drawImage(0, 0, self.overlay_qimg)

        # 커서(화면 좌표계)
        if self.cursor_pos is not None:
            painter.resetTransform()
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            if self.mode in ("brush", "erase"):
                r = int(self.brush_size * self.scale_factor)
            else:
                r = 8  # SAM 클릭 커서 고정
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

        # buttons
        btn_brush = QPushButton("Brush")
        btn_erase = QPushButton("Eraser")
        btn_sam   = QPushButton("SAM Brush")
        btn_prev  = QPushButton("Prev")
        btn_next  = QPushButton("Next")

        btn_brush.clicked.connect(lambda: self.canvas.set_mode("brush"))
        btn_erase.clicked.connect(lambda: self.canvas.set_mode("erase"))
        btn_sam.clicked.connect(lambda: self.canvas.set_mode("sam"))
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)

        # brush size slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(80)
        slider.setValue(10)
        slider.valueChanged.connect(self.canvas.set_brush_size)

        # layout
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout()
        root.setLayout(layout)
        layout.addWidget(self.scroll)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_brush)
        btn_layout.addWidget(btn_erase)
        btn_layout.addWidget(btn_sam)
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)
        layout.addLayout(btn_layout)
        layout.addWidget(slider)

        # Ctrl+Z Undo
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        self.load_current()

    def auto_save(self):
        mask = self.canvas.get_mask()
        if mask is None or self.current_name is None:
            return
        # 예: current_name = "000001_obj0_mask" -> "000001_obj0_mask_edit.png"
        save_path = AFTER_DIR / f"{self.current_name}_edit.png"
        cv2.imwrite(str(save_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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