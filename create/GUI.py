import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from collections import OrderedDict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QToolButton,
    QVBoxLayout, QHBoxLayout, QSlider, QScrollArea, QListWidget,
    QListWidgetItem, QFileDialog, QLabel, QSplitter, QMessageBox,
    QButtonGroup, QSizePolicy
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeySequence, QShortcut,
    QPainter, QPen, QColor, QFont
)
from PySide6.QtCore import Qt, QTimer, QRect

from segment_anything import sam_model_registry, SamPredictor


# ======================
# Paths ###################################### 본인 BASE_DIR 입력력
# ======================
#BASE_DIR = Path(r"C:\Users\user\Desktop\pj\research\data_annotation_sw\create")



IMAGE_DIR = BASE_DIR / "input"
MASK_DIR  = BASE_DIR / "output_1" / "masks"
LABEL_DIR = BASE_DIR / "output_1" / "labels"

AFTER_IMAGE_DIR = BASE_DIR / "output_2" / "images"
AFTER_MASK_DIR  = BASE_DIR / "output_2" / "masks"
AFTER_LABEL_DIR = BASE_DIR / "output_2" / "labels"

AFTER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
AFTER_MASK_DIR.mkdir(parents=True, exist_ok=True)
AFTER_LABEL_DIR.mkdir(parents=True, exist_ok=True)

SAM_WEIGHT = r"C:\Users\user\Desktop\pj\research\data_annotation_sw\weights\sam_vit_b_01ec64.pth"


# ======================
# Label utils
# ======================

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


def parse_instance_id_from_stem(stem: str) -> str:
    name = stem
    for suf in ("_edited", "_edit", "_masked", "_mask"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if name.endswith("_mask"):
        name = name[: -len("_mask")]
    return name


# ======================
# SAM
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHT))
sam.to(device)
predictor = SamPredictor(sam)


# ======================
# Small LRU cache
# ======================
class LRUCache:
    def __init__(self, capacity: int = 16):
        self.capacity = int(capacity)
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


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

        self.on_modified = None
        self.on_need_sam_ready = None

    def notify_modified(self):
        if callable(self.on_modified):
            self.on_modified()

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
        if mode != "sam":
            self.sam_points = []
            self.sam_labels = []
        self.update()

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
        self.notify_modified()

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
            self.notify_modified()
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
                self.notify_modified()
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

            if callable(self.on_need_sam_ready) and not self.on_need_sam_ready():
                return

            self.push_undo()
            self.run_sam()
            self.notify_modified()
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
        self.setWindowTitle("Data Annotation - Mask Editor")
        self.resize(1600, 900)

        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fa; }
            QPushButton, QToolButton {
                font-size: 14px;
                font-weight: bold;
                padding: 10px 18px;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 6px;
                color: #495057;
                min-height: 24px;
            }
            QPushButton:hover, QToolButton:hover { background-color: #e9ecef; }
            QToolButton:checked {
                background-color: #e7f5ff;
                border: 2px solid #339af0;
                color: #1864ab;
            }
            QListWidget {
                font-size: 13px;
                border: 1px solid #ced4da;
                border-radius: 6px;
                background-color: #ffffff;
                padding: 5px;
            }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #f1f3f5; }
            QListWidget::item:selected {
                background-color: #d0ebff;
                color: #0b7285;
                border-radius: 4px;
            }
            QScrollArea {
                border: 1px solid #ced4da;
                border-radius: 6px;
                background-color: #e9ecef;
            }
            QLabel { font-size: 14px; color: #495057; font-weight: bold; }
        """)

        self.image_files = self.list_input_images()
        if not self.image_files:
            QMessageBox.critical(self, "No images", f"No image files found in:\n{IMAGE_DIR}")
            sys.exit(0)

        self.index = -1
        self.current_mask_path: Path | None = None
        self.current_stem: str | None = None
        self.current_image_key: str | None = None

        self.baseline_mask: np.ndarray | None = None
        self.dirty = False

        self.image_cache = LRUCache(capacity=24)
        self.mask_cache = LRUCache(capacity=48)

        self.sam_ready_for_image_key: str | None = None

        self.canvas = UnifiedCanvas()
        self.canvas.on_modified = self.recompute_dirty
        self.canvas.on_need_sam_ready = self.ensure_sam_ready_for_current_image

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(300)
        self.list_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.populate_list()
        self.list_widget.currentRowChanged.connect(self.on_list_row_changed)

        self.btn_brush = QToolButton()
        self.btn_brush.setText("🖌️ Brush")
        self.btn_brush.setCheckable(True)

        self.btn_erase = QToolButton()
        self.btn_erase.setText("🧽 Eraser")
        self.btn_erase.setCheckable(True)

        self.btn_sam = QToolButton()
        self.btn_sam.setText("🪄 SAM")
        self.btn_sam.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_brush)
        self.mode_group.addButton(self.btn_erase)
        self.mode_group.addButton(self.btn_sam)
        self.btn_brush.setChecked(True)

        self.btn_brush.clicked.connect(self.on_click_brush)
        self.btn_erase.clicked.connect(self.on_click_erase)
        self.btn_sam.clicked.connect(self.on_click_sam)

        self.btn_open = QPushButton("📂 Open")
        self.btn_save = QPushButton("💾 Save")
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")

        self.btn_open.clicked.connect(self.open_file_dialog)
        self.btn_save.clicked.connect(self.save_if_dirty_force)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(80)
        self.slider.setValue(10)
        self.slider.setFixedWidth(150)
        self.slider.valueChanged.connect(self.canvas.set_brush_size)

        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(280)
        self.status_label.setStyleSheet("color: #868e96; font-weight: normal;")

        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(10, 15, 10, 10)
        bottom_bar.addWidget(self.btn_open)
        bottom_bar.addSpacing(15)
        bottom_bar.addWidget(self.btn_brush)
        bottom_bar.addWidget(self.btn_erase)
        bottom_bar.addWidget(self.btn_sam)
        bottom_bar.addSpacing(20)
        bottom_bar.addWidget(QLabel("Brush Size:"))
        bottom_bar.addWidget(self.slider)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addSpacing(20)
        bottom_bar.addWidget(self.btn_save)
        bottom_bar.addWidget(self.btn_prev)
        bottom_bar.addWidget(self.btn_next)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.addWidget(splitter, 1)
        layout.addLayout(bottom_bar)
        root.setLayout(layout)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_if_dirty_force)

        self.next_shortcut = QShortcut(QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.next_image)

        self.prev_shortcut = QShortcut(QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.prev_image)

        self.brush_shortcut = QShortcut(QKeySequence("1"), self)
        self.brush_shortcut.activated.connect(self.on_click_brush)

        self.erase_shortcut = QShortcut(QKeySequence("2"), self)
        self.erase_shortcut.activated.connect(self.on_click_erase)

        self.sam_shortcut = QShortcut(QKeySequence("3"), self)
        self.sam_shortcut.activated.connect(self.on_click_sam)

        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.activated.connect(lambda: self.scroll_view(-80, 0))

        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.activated.connect(lambda: self.scroll_view(80, 0))

        self.up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.up_shortcut.activated.connect(lambda: self.scroll_view(0, -80))

        self.down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.down_shortcut.activated.connect(lambda: self.scroll_view(0, 80))

        self.list_widget.setCurrentRow(0)

    # ---------- File list ----------
    def list_input_images(self):
        files = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
            files.extend(IMAGE_DIR.glob(ext))
        return sorted(files)

    # ---------- Mode handlers ----------
    def on_click_brush(self):
        self.canvas.set_mode("brush")
        self.set_status("Brush mode")

    def on_click_erase(self):
        self.canvas.set_mode("erase")
        self.set_status("Eraser mode")

    def on_click_sam(self):
        self.canvas.set_mode("sam")
        ok = self.ensure_sam_ready_for_current_image()
        if ok:
            self.set_status("SAM mode (ready)")
        else:
            self.btn_brush.setChecked(True)
            self.canvas.set_mode("brush")

    def scroll_view(self, dx, dy):
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(hbar.value() + dx)
        vbar.setValue(vbar.value() + dy)

    def move_remaining_files(self):
        moved = 0

        image_files = self.list_input_images()

        for image_path in image_files:
            stem = image_path.stem

            edited_mask = AFTER_MASK_DIR / f"{stem}_edited.png"
            if edited_mask.exists():
                continue

            base_mask = MASK_DIR / f"{stem}.png"
            base_label = LABEL_DIR / f"{stem}.txt"

            dst_image = AFTER_IMAGE_DIR / f"{stem}.png"
            dst_mask = AFTER_MASK_DIR / f"{stem}.png"
            dst_label = AFTER_LABEL_DIR / f"{stem}.txt"

            try:
                img = cv2.imread(str(image_path))
                if img is not None:
                    cv2.imwrite(str(dst_image), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                if base_mask.exists():
                    mask = cv2.imread(str(base_mask), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        cv2.imwrite(str(dst_mask), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                if base_label.exists():
                    with open(base_label, "r", encoding="utf-8") as fsrc, open(dst_label, "w", encoding="utf-8") as fdst:
                        fdst.write(fsrc.read())

                moved += 1

            except Exception as e:
                print("Move error:", e)

        print(f"Moved remaining files: {moved}")

    # ---------- Dirty handling ----------
    def recompute_dirty(self):
        if self.baseline_mask is None:
            return
        cur = self.canvas.get_mask()
        if cur is None:
            return

        is_dirty = not np.array_equal(cur, self.baseline_mask)
        if is_dirty != self.dirty:
            self.dirty = is_dirty
            if self.dirty:
                self.set_status("⚠️ Modified (not saved)")
                self.status_label.setStyleSheet("color: #e03131; font-weight: bold;")
            else:
                self.set_status("Ready")
                self.status_label.setStyleSheet("color: #868e96; font-weight: normal;")

    def set_baseline_from_current(self):
        m = self.canvas.get_mask()
        self.baseline_mask = None if m is None else m.copy()
        self.dirty = False
        self.status_label.setStyleSheet("color: #868e96; font-weight: normal;")

    # ---------- Status / Cursor ----------
    def set_status(self, msg: str):
        self.status_label.setText(msg)

    def run_with_wait_cursor(self, status_msg: str, fn):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.set_status(status_msg)
            QApplication.processEvents()
            return fn()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    # ---------- Save Prompt ----------
    def check_unsaved_changes(self) -> bool:
        if not self.dirty:
            return True

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Unsaved Changes")
        msg_box.setText("There are unsaved changes. Do you want to save them?")
        msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Save)

        ret = msg_box.exec()

        if ret == QMessageBox.Save:
            self._save_current()
            return True
        elif ret == QMessageBox.Cancel:
            return False
        else:
            self.set_baseline_from_current()
            self.set_status("Discarded changes")
            return True

    def closeEvent(self, event):
        if not self.check_unsaved_changes():
            event.ignore()
            return

        reply = QMessageBox.question(
            self,
            "작업 종료",
            "수정하지 않은 이미지도 output_2로 이동하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.move_remaining_files()
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

    # ---------- UI list ----------
    def populate_list(self):
        self.list_widget.clear()
        for p in self.image_files:
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.list_widget.addItem(item)

    def on_list_row_changed(self, row: int):
        if row < 0 or row >= self.list_widget.count():
            return
        if row == self.index and self.current_image_key is not None:
            return

        if not self.check_unsaved_changes():
            self.list_widget.blockSignals(True)
            self.list_widget.setCurrentRow(self.index if self.index >= 0 else 0)
            self.list_widget.blockSignals(False)
            return

        item = self.list_widget.item(row)
        path = Path(item.data(Qt.UserRole))
        self.index = row
        self.load_image_path(path)

    def open_file_dialog(self):
        start_dir = str(IMAGE_DIR)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        if not self.check_unsaved_changes():
            return

        p = Path(file_path)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if Path(item.data(Qt.UserRole)) == p:
                self.list_widget.setCurrentRow(i)
                return

        self.load_image_path(p)
        self.set_status(f"Opened (external): {p.name}")

    # ---------- Loading ----------
    def find_image_path(self, image_key: str) -> Path | None:
        for ext in ("jpg", "png", "jpeg", "bmp"):
            cand = IMAGE_DIR / f"{image_key}.{ext}"
            if cand.exists():
                return cand
        return None

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

    def _read_image_cached(self, image_path: Path):
        key = str(image_path)
        cached = self.image_cache.get(key)
        if cached is not None:
            return cached
        img_bgr = cv2.imread(key)
        if img_bgr is not None:
            self.image_cache.put(key, img_bgr)
        return img_bgr

    def _read_mask_cached(self, mask_path: Path):
        key = str(mask_path)
        cached = self.mask_cache.get(key)
        if cached is not None:
            return cached.copy()
        mask = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask = (mask > 127).astype(np.uint8) * 255
        self.mask_cache.put(key, mask.copy())
        return mask

    def fit_image_to_view(self):
        if self.canvas.image_rgb is None:
            return

        img_h, img_w = self.canvas.image_rgb.shape[:2]
        view_w = self.scroll.viewport().width()
        view_h = self.scroll.viewport().height()

        if view_w <= 0 or view_h <= 0:
            return

        scale_w = view_w / img_w
        scale_h = view_h / img_h

        self.canvas.scale_factor = min(scale_w, scale_h) * 0.9
        self.canvas.update_widget_size()
        self.canvas.update()

        self.focus_on_mask()

    def load_image_path(self, image_path: Path):
        if not image_path.exists():
            QMessageBox.warning(self, "Missing file", f"Image not found:\n{image_path}")
            return

        stem = image_path.stem
        image_key = stem

        edited_mask_path = AFTER_MASK_DIR / f"{stem}_edited.png"
        base_mask_path = MASK_DIR / f"{stem}.png"

        if edited_mask_path.exists():
            mask_to_load = edited_mask_path
        elif base_mask_path.exists():
            mask_to_load = base_mask_path
        else:
            mask_to_load = None

        img_bgr = self._read_image_cached(image_path)
        if img_bgr is None:
            QMessageBox.warning(self, "Read fail", f"Failed to read image:\n{image_path}")
            return

        h, w = img_bgr.shape[:2]

        if mask_to_load is not None:
            mask = self._read_mask_cached(mask_to_load)
            if mask is None:
                QMessageBox.warning(self, "Read fail", f"Failed to read mask:\n{mask_to_load}")
                return
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        self.current_mask_path = base_mask_path
        self.current_stem = stem
        self.current_image_key = image_key

        self.sam_ready_for_image_key = None

        self.canvas.set_data(img_bgr, mask)
        self.set_baseline_from_current()

        if edited_mask_path.exists():
            self.set_status(f"Loaded (edited): {stem}")
        elif base_mask_path.exists():
            self.set_status(f"Loaded: {stem}")
        else:
            self.set_status(f"Loaded (new / empty mask): {stem}")

        QTimer.singleShot(0, self.fit_image_to_view)

    # ---------- SAM lazy embedding ----------
    def ensure_sam_ready_for_current_image(self) -> bool:
        if self.current_image_key is None:
            return False
        if self.sam_ready_for_image_key == self.current_image_key:
            return True

        img_bgr = self.canvas.image_bgr
        if img_bgr is None:
            return False

        def _do():
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)
            self.sam_ready_for_image_key = self.current_image_key
            return True

        ok = self.run_with_wait_cursor("⏳ Preparing SAM embedding...", _do)
        return bool(ok)

    # ---------- Saving ----------
    def save_if_dirty_force(self):
        if self.current_stem is None:
            return
        if not self.dirty:
            self.set_status("No changes")
            return
        self._save_current()

    def _save_current(self):
        mask = self.canvas.get_mask()
        if mask is None or self.current_stem is None:
            return

        stem = self.current_stem

        image_save_path = AFTER_IMAGE_DIR / f"{stem}.png"
        if self.canvas.image_bgr is not None:
            cv2.imwrite(str(image_save_path), self.canvas.image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        parts = stem.split("_")
        try:
            if parts[0].isnumeric():
                class_id = int(parts[0]) - 1
            else:
                class_id = int(parts[1]) - 1
        except Exception:
            QMessageBox.warning(self, "Filename error", f"Cannot infer class id from filename:\n{stem}")
            return

        save_mask_path = AFTER_MASK_DIR / f"{stem}_edited.png"
        ok = cv2.imwrite(str(save_mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            QMessageBox.warning(self, "Save failed", f"Failed to write:\n{save_mask_path}")
            return

        bbox, poly = mask_to_bbox_and_polygon(mask)
        if bbox is None or poly is None or len(poly) < 3:
            QMessageBox.warning(self, "Invalid mask", f"Empty/invalid mask:\n{stem}")
            return

        h, w = mask.shape
        poly_norm = []

        for px, py in poly:
            xn = px / (w - 1)
            yn = py / (h - 1)
            poly_norm.append(f"{xn:.6f}")
            poly_norm.append(f"{yn:.6f}")

        line = str(class_id) + " " + " ".join(poly_norm)

        dst_label_path = AFTER_LABEL_DIR / f"{stem}.txt"
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")

        self.set_baseline_from_current()
        self.set_status(f"✅ Saved: {stem}")

    # ---------- Navigation ----------
    def next_image(self):
        if self.dirty:
            if not self.check_unsaved_changes():
                return

        if self.index < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(self.index + 1)
        else:
            self.close()

    def prev_image(self):
        if self.index > 0:
            self.list_widget.setCurrentRow(self.index - 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    win = UnifiedEditor()
    win.showMaximized()
    sys.exit(app.exec())