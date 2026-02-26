import sys
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QVBoxLayout, QWidget, QHBoxLayout,
    QSlider, QScrollArea
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut,
    QPainter, QPen, QColor
)
from PySide6.QtCore import Qt, QTimer, QRect


# ======================
# 경로 설정 (images/* 구조 사용)
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # .../data_annotation
IMAGE_DIR = BASE_DIR / "images" / "original"
MASK_DIR  = BASE_DIR / "images" / "yolo_seg_results" / "masks"
AFTER_DIR = BASE_DIR / "images" / "edit_results" / "masks"
AFTER_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# Canvas (Fast)
# ======================
class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)

        self.image_rgb = None            # np uint8 (H,W,3) RGB
        self.mask = None                 # np uint8 (H,W) 0/255

        self.image_qimg = None           # QImage (wraps numpy)
        self.image_pix = None            # QPixmap

        # 오버레이 캐시 (ARGB)
        self.overlay_np = None           # np uint8 (H,W,4)
        self.overlay_qimg = None         # QImage (wraps overlay_np)

        self.scale_factor = 1.0
        self.brush_size = 10
        self.mode = "brush"              # "brush" or "erase"
        self.drawing = False

        # stroke 보간
        self.last_xy = None

        # Undo
        self.mask_history = []
        self.max_undo = 30

        # cursor 표시
        self.cursor_pos = None

        # overlay alpha
        self.overlay_alpha = 110

    def set_data(self, image_rgb, mask_u8):
        self.image_rgb = np.ascontiguousarray(image_rgb)
        self.mask = np.ascontiguousarray(mask_u8)

        h, w, _ = self.image_rgb.shape

        # 이미지 QImage/QPixmap (메모리 유지 위해 numpy를 멤버로 보관)
        self.image_qimg = QImage(self.image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.image_pix = QPixmap.fromImage(self.image_qimg)

        # 오버레이 캐시 준비
        self.overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
        self.overlay_np[..., 1] = 255  # Green 채널 고정 (알파=0인 곳은 영향 없음)
        self.overlay_qimg = QImage(self.overlay_np.data, w, h, 4 * w, QImage.Format_ARGB32)

        # 초기 오버레이 전체 갱신
        self.rebuild_overlay_full()

        # 스케일/언두 리셋
        self.scale_factor = 1.0
        self.mask_history = []
        self.last_xy = None

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
        # alpha = mask>0 ? overlay_alpha : 0
        alpha = np.where(self.mask > 0, self.overlay_alpha, 0).astype(np.uint8)
        self.overlay_np[..., 3] = alpha

    def update_overlay_dirty(self, x1, y1, x2, y2):
        """x2,y2 inclusive"""
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

        # repaint는 스케일 좌표로 dirty만
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
        if self.drawing:
            self.paint_to_mask(event)
        else:
            self.update()  # cursor만
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.image_rgb is None or self.mask is None:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.push_undo()
            self.last_xy = None
            self.paint_to_mask(event)

    def mouseReleaseEvent(self, event):
        self.drawing = False
        self.last_xy = None

    def paint_to_mask(self, event):
        x = int(event.position().x() / self.scale_factor)
        y = int(event.position().y() / self.scale_factor)

        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        r = int(self.brush_size)
        color = 255 if self.mode == "brush" else 0

        # stroke 보간: 마지막 점과 현재 점 사이를 선으로 채움
        if self.last_xy is None:
            cv2.circle(self.mask, (x, y), r, int(color), -1)
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            self.update_overlay_dirty(x1, y1, x2, y2)
            self.last_xy = (x, y)
            return

        x0, y0 = self.last_xy
        # 선 두께 = 2r 정도
        cv2.line(self.mask, (x0, y0), (x, y), int(color), thickness=max(1, 2 * r))
        # 끝점 원도 추가 (모서리 매끈)
        cv2.circle(self.mask, (x, y), r, int(color), -1)

        x1 = min(x0, x) - r - 2
        y1 = min(y0, y) - r - 2
        x2 = max(x0, x) + r + 2
        y2 = max(y0, y) + r + 2
        self.update_overlay_dirty(x1, y1, x2, y2)

        self.last_xy = (x, y)

    def paintEvent(self, event):
        if self.image_pix is None or self.overlay_qimg is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 이미지 좌표계로 스케일
        painter.scale(self.scale_factor, self.scale_factor)

        # 1) 원본
        painter.drawPixmap(0, 0, self.image_pix)

        # 2) 마스크 오버레이 (캐시)
        painter.drawImage(0, 0, self.overlay_qimg)

        # 3) 커서 (화면 좌표계로 그려야 함)
        if self.cursor_pos is not None:
            painter.resetTransform()
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            r = int(self.brush_size * self.scale_factor)
            painter.drawEllipse(self.cursor_pos, r, r)


# ======================
# Main Window
# ======================
class MaskEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Editor (Brush/Eraser) - Fast")
        self.resize(1800, 1000)

        self.mask_files = sorted(MASK_DIR.glob("*.png"))
        if not self.mask_files:
            print("No mask files found in:", MASK_DIR)
            sys.exit(0)

        self.index = 0
        self.current_name = None

        self.canvas = Canvas()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)

        # buttons
        btn_brush = QPushButton("Brush")
        btn_erase = QPushButton("Eraser")
        btn_prev  = QPushButton("Prev")
        btn_next  = QPushButton("Next")

        btn_brush.clicked.connect(lambda: self.set_mode("brush"))
        btn_erase.clicked.connect(lambda: self.set_mode("erase"))
        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)

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

    def auto_save(self):
        mask = self.canvas.get_mask()
        if mask is None or self.current_name is None:
            return
        # 예: current_name = "000001_obj0_mask" -> "000001_obj0_mask_edit.png"
        save_path = AFTER_DIR / f"{self.current_name}_edit.png"
        # 저장 끊김 줄이기: 압축 낮춤(0~9, 낮을수록 빠름)
        cv2.imwrite(str(save_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Failed to read mask:", mask_path)
            return
        mask = (mask > 127).astype(np.uint8) * 255

        self.current_name = stem
        self.canvas.set_data(image_rgb, mask)

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