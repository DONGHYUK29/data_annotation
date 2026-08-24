import os
import sys
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QWidget, QHBoxLayout,
    QSlider, QLabel, QScrollArea
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut,
    QPainter, QPen, QColor
)
from PySide6.QtCore import Qt, QTimer


# ======================
# 경로 설정 (SAM과 동일)
# ======================

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(
    os.environ.get("KERI_PIPELINE_GUI_WORKDIR", REPO_ROOT / "data" / "pipeline_gui")
).expanduser().resolve()
IMAGE_DIR = BASE_DIR / "before/images"
MASK_DIR  = BASE_DIR / "before/masks"
AFTER_DIR = BASE_DIR / "after/masks"
AFTER_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# Canvas
# ======================

class Canvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)

        self.image = None              # RGB
        self.mask = None               # uint8 0/255
        self.pixmap_original = None

        self.scale_factor = 1.0
        self.brush_size = 10
        self.mode = "brush"            # "brush" or "erase"
        self.drawing = False

        # Undo
        self.mask_history = []

        # cursor 표시
        self.cursor_pos = None
        self.cursor_radius = 8

    def set_data(self, image_rgb, mask_u8):
        self.image = image_rgb
        self.mask = mask_u8
        self.scale_factor = 1.0
        self.mask_history = []
        self.update_display()

    def get_mask(self):
        return self.mask

    def push_undo(self):
        if self.mask is not None:
            self.mask_history.append(self.mask.copy())

    def undo(self):
        if len(self.mask_history) > 0:
            self.mask = self.mask_history.pop()
            self.update_display()

    def update_display(self):
        if self.image is None or self.mask is None:
            return

        overlay = self.image.copy().astype(np.float32)
        green = np.array([0, 255, 0], dtype=np.float32)

        # mask overlay
        overlay[self.mask > 0] = (
            0.6 * overlay[self.mask > 0] +
            0.4 * green
        )
        overlay = overlay.astype(np.uint8)

        h, w, ch = overlay.shape
        bytes_per_line = ch * w

        qimg = QImage(
            overlay.data, w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        self.pixmap_original = QPixmap.fromImage(qimg)
        self.apply_scale()

    def apply_scale(self):
        if self.pixmap_original is None:
            return

        new_w = int(self.pixmap_original.width() * self.scale_factor)
        new_h = int(self.pixmap_original.height() * self.scale_factor)

        scaled = self.pixmap_original.scaled(
            new_w, new_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.resize(scaled.size())

    # Ctrl+휠 확대/축소, 일반 휠은 스크롤로 넘김
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor *= 0.9
            self.scale_factor = max(0.3, min(self.scale_factor, 5.0))
            self.apply_scale()
        else:
            event.ignore()  # ✅ ScrollArea가 wheel을 처리하도록

    # cursor 위치 추적
    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position().toPoint()
        self.update()

        if self.drawing:
            self.paint(event)

    def mousePressEvent(self, event):
        if self.image is None or self.mask is None:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.push_undo()   # ✅ stroke 시작 시 undo 스냅샷 저장
            self.paint(event)

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def paint(self, event):
        if self.image is None or self.mask is None:
            return

        x = int(event.position().x() / self.scale_factor)
        y = int(event.position().y() / self.scale_factor)

        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        color = 255 if self.mode == "brush" else 0

        cv2.circle(
            self.mask,
            (x, y),
            int(self.brush_size),
            int(color),
            -1
        )
        self.update_display()

    # 빨간 원 커서 표시
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.cursor_pos is None:
            return

        painter = QPainter(self)

        # 🔴 빨간 테두리
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)

        # ✅ 실제 브러시 반경 = brush_size × scale
        r = int(self.brush_size * self.scale_factor)

        painter.drawEllipse(
            self.cursor_pos.x() - r,
            self.cursor_pos.y() - r,
            2 * r,
            2 * r
        )



# ======================
# Main Window
# ======================

class MaskEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Editor (Brush/Eraser)")
        self.resize(1800, 1000)

        self.mask_files = sorted(MASK_DIR.glob("*.png"))
        if not self.mask_files:
            print("No mask files found in:", MASK_DIR)
            sys.exit(0)

        self.index = 0
        self.canvas = Canvas()

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)  # ✅ 스크롤 정확히 쓰기

        # buttons
        btn_brush = QPushButton("Brush")
        btn_erase = QPushButton("Eraser")
        btn_prev  = QPushButton("Prev")
        btn_next  = QPushButton("Next")

        btn_brush.clicked.connect(lambda: self.set_mode("brush"))
        btn_erase.clicked.connect(lambda: self.set_mode("erase"))
        btn_prev.clicked.connect(self.prev_image)   # ✅ Prev도 자동 저장
        btn_next.clicked.connect(self.next_image)   # ✅ Next도 자동 저장

        # brush size slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(80)
        slider.setValue(10)
        slider.valueChanged.connect(self.set_brush_size)

        # layout
        layout = QVBoxLayout()
        layout.addWidget(self.scroll)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_brush)
        btn_layout.addWidget(btn_erase)
        btn_layout.addWidget(btn_prev)
        btn_layout.addWidget(btn_next)
        layout.addLayout(btn_layout)
        layout.addWidget(slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Ctrl+Z Undo
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        self.load_current()

    # ✅ 자동 저장
    def auto_save(self):
        mask = self.canvas.get_mask()
        if mask is None:
            return
        save_path = AFTER_DIR / f"{self.current_name}.png"
        cv2.imwrite(str(save_path), mask)

    # ✅ 마스크 bbox 중심으로 포커스
    def focus_on_mask(self):
        mask = self.canvas.get_mask()
        if mask is None:
            return

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # scale 반영
        cx = int(cx * self.canvas.scale_factor)
        cy = int(cy * self.canvas.scale_factor)

        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()

        h_bar.setValue(max(0, cx - self.scroll.viewport().width() // 2))
        v_bar.setValue(max(0, cy - self.scroll.viewport().height() // 2))

    def load_current(self):
        mask_path = self.mask_files[self.index]
        stem = mask_path.stem              # e.g. 2_obj0
        image_key = stem.split("_obj")[0]  # e.g. 2

        # image 확장자 탐색
        image_path = None
        for ext in ["jpg", "png", "jpeg"]:
            cand = IMAGE_DIR / f"{image_key}.{ext}"
            if cand.exists():
                image_path = cand
                break

        if image_path is None:
            print("Image not found for:", stem)
            return

        # read
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print("Failed to read image:", image_path)
            return
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Failed to read mask:", mask_path)
            return
        mask = (mask > 127).astype(np.uint8) * 255

        self.current_name = stem
        self.canvas.set_data(image_rgb, mask)

        # ✅ 스크롤 range 계산 이후 포커스 (지연 실행)
        QTimer.singleShot(0, self.focus_on_mask)

    def set_mode(self, mode):
        self.canvas.mode = mode

    def set_brush_size(self, value):
        self.canvas.brush_size = int(value)

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
    editor = MaskEditor()
    editor.showMaximized()
    sys.exit(app.exec())
