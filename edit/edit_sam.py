import sys
import cv2
import numpy as np
from pathlib import Path
import torch

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut,
    QPainter, QPen, QColor
)
from PySide6.QtCore import Qt, QTimer

from segment_anything import sam_model_registry, SamPredictor


# ======================
# 경로 설정 (images/* 구조 사용)
# ======================

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # .../data_annotation
BEFORE_IMG_DIR = BASE_DIR / "images" / "original"
BEFORE_MASK_DIR = BASE_DIR / "images" / "yolo_seg_results" / "masks"
AFTER_MASK_DIR = BASE_DIR / "images" / "edit_results" / "masks"
SAM_WEIGHT = BASE_DIR / "weights/sam_vit_b_01ec64.pth"

AFTER_MASK_DIR.mkdir(parents=True, exist_ok=True)

###########################################################################경로 및 파일명 수정 -> 파일 정리

# ======================
# SAM 로드
# ======================

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHT))
sam.to(device)
predictor = SamPredictor(sam)


# ======================
# Canvas
# ======================

class SamCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)

        self.image = None
        self.mask = None
        self.mask_history = []

        self.points = []
        self.labels = []

        self.scale_factor = 1.0

        # 마우스 표시용
        self.cursor_pos = None
        self.cursor_radius = 8

    def load_data(self, image, mask):
        self.image = image
        self.mask = mask
        self.mask_history = []
        self.points = []
        self.labels = []
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.update_display()

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position().toPoint()
        self.update()

    def mousePressEvent(self, event):
        if self.image is None:
            return

        x = int(event.position().x() / self.scale_factor)
        y = int(event.position().y() / self.scale_factor)

        if event.button() == Qt.LeftButton:
            self.points.append([x, y])
            self.labels.append(1)
        elif event.button() == Qt.RightButton:
            self.points.append([x, y])
            self.labels.append(0)

        self.run_sam()

    def run_sam(self):
        if len(self.points) == 0:
            return

        if self.mask is not None:
            self.mask_history.append(self.mask.copy())

        input_points = np.array(self.points)
        input_labels = np.array(self.labels)

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

        self.mask = (masks[0] * 255).astype("uint8")
        self.update_display()

    def undo(self):
        if len(self.mask_history) > 0:
            self.mask = self.mask_history.pop()
            self.update_display()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor *= 0.9

            self.scale_factor = max(0.3, min(self.scale_factor, 5.0))
            self.update_display()
        else:
            super().wheelEvent(event)

    def update_display(self):
        if self.image is None:
            return

        overlay = self.image.copy()

        if self.mask is not None:
            overlay[self.mask > 0] = (
                0.6 * overlay[self.mask > 0] +
                0.4 * np.array([0, 255, 0])
            ).astype("uint8")

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            int(w * self.scale_factor),
            int(h * self.scale_factor),
            Qt.KeepAspectRatio,111111111111
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)
        self.resize(scaled.size())

    def get_mask(self):
        return self.mask

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.cursor_pos is None:
            return

        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)

        r = self.cursor_radius
        painter.drawEllipse(
            self.cursor_pos.x() - r,
            self.cursor_pos.y() - r,
            2 * r,
            2 * r
        )


# ======================
# Main Window
# ======================

class SamEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Refine Tool")
        self.resize(1800, 1000)

        self.canvas = SamCanvas()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)

        # 중요: 스크롤 계산을 위해 False 권장 (캔버스가 실제 크기를 갖고 스크롤바가 생김)
        self.scroll.setWidgetResizable(False)

        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        self.files = sorted(BEFORE_MASK_DIR.glob("*.png"))
        self.index = 0

        if not self.files:
            print("No masks found.")
            return

        self.load_current()

    def center_scroll(self):
        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()
        h_bar.setValue(h_bar.maximum() // 2)
        v_bar.setValue(v_bar.maximum() // 2)

    def load_current(self):
        mask_path = self.files[self.index]
        stem = mask_path.stem

        image_name = stem.split("_obj")[0]

        # jpg 우선, 없으면 png/jpeg도 탐색
        image_path = None
        for ext in ["jpg", "png", "jpeg"]:
            cand = BEFORE_IMG_DIR / f"{image_name}.{ext}"
            if cand.exists():
                image_path = cand
                break

        if image_path is None:
            print("Missing image for:", stem)
            return

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), 0)

        if image is None or mask is None:
            print("Missing file:", stem)
            return

        self.current_name = stem
        self.canvas.load_data(image, mask)

        # ✅ 핵심: 스크롤바 range 계산이 끝난 다음 중앙 이동 (지연 실행)
        QTimer.singleShot(0, self.focus_on_mask)

    def auto_save(self):
        mask = self.canvas.get_mask()
        if mask is None:
            return
        # 예: current_name = "000001_obj0_mask" -> "000001_obj0_mask_edit.png"
        save_path = AFTER_MASK_DIR / f"{self.current_name}_edit.png"
        cv2.imwrite(str(save_path), mask)

    def next_image(self):
        self.auto_save()
        if self.index < len(self.files) - 1:
            self.index += 1
            self.load_current()
        else:
            self.close()

    def prev_image(self):
        self.auto_save()
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def focus_on_mask(self):
        mask = self.canvas.get_mask()
        if mask is None:
            return

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return

        # 🔥 bbox 계산
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # bbox 중심
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 스케일 반영
        cx = int(cx * self.canvas.scale_factor)
        cy = int(cy * self.canvas.scale_factor)

        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()

        # 화면 중앙에 오도록 이동
        h_bar.setValue(max(0, cx - self.scroll.viewport().width() // 2))
        v_bar.setValue(max(0, cy - self.scroll.viewport().height() // 2))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SamEditor()
    window.showMaximized()
    window.show()
    sys.exit(app.exec())
