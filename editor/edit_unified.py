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
    name = stem
    for suf in ("_edited", "_edit", "_masked", "_mask"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if name.endswith("_mask"):
        name = name[: -len("_mask")]
    return name

def find_src_label_path(image_id: str) -> Path:
    p = LABEL_DIR / f"{image_id}_detected.txt"
    if p.exists():
        return p
    p2 = LABEL_DIR / f"{image_id}.txt"
    return p2

def build_class_map_from_src_lines(src_lines: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    fallback_cls: int | None = None

    for ln in src_lines:
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            cls = int(float(parts[1]))
        except ValueError:
            continue

        if "_obj" in key:
            mapping[key] = cls
        else:
            if fallback_cls is None:
                fallback_cls = cls

    if mapping:
        return mapping

    if fallback_cls is not None:
        return {"*": fallback_cls}

    return {}

def get_class_id_for_instance(class_map: dict[str, int], instance_id: str) -> int | None:
    if instance_id in class_map:
        return class_map[instance_id]
    if "*" in class_map:
        return class_map["*"]
    return None


# ======================
# SAM
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHT))
sam.to(device)
predictor = SamPredictor(sam)


# ======================
# Small LRU cache (for faster switching)
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
    """
    - 이미지/마스크 렌더링 + 브러시/지우개 + SAM 포인트 입력
    - SAM embedding(set_image)은 에디터가 준비(ensure_sam_ready)해준 뒤에만 run_sam 실행
    """
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
        self.on_need_sam_ready = None  # editor callback

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

            # SAM은 클릭 시마다 run_sam 필요 -> 그 전에 embedding 준비되어야 함
            if callable(self.on_need_sam_ready) and not self.on_need_sam_ready():
                # 준비 실패/취소
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
    """
    핵심 변경점:
    - 파일 이동 시 predictor.set_image()를 호출하지 않음 (빠른 전환)
    - SAM 모드 버튼을 눌렀을 때(또는 SAM 첫 클릭 시)만 set_image() 수행
    - set_image 수행 중에는 WaitCursor(로딩 커서) + 상태 메시지 표시
    - dirty는 baseline mask와 현재 mask 비교로 판단 (Ctrl+Z로 원상복귀하면 dirty 해제)
    """
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

        self.mask_files = sorted(MASK_DIR.glob("*.png"))
        if not self.mask_files:
            QMessageBox.critical(self, "No masks", f"No mask files found in:\n{MASK_DIR}")
            sys.exit(0)

        # start -1 so rowChanged(0) loads at startup
        self.index = -1
        self.current_mask_path: Path | None = None
        self.current_stem: str | None = None

        # track current original image key for this mask
        self.current_image_key: str | None = None

        # baseline dirty compare
        self.baseline_mask: np.ndarray | None = None
        self.dirty = False

        # caches (disk decode)
        self.image_cache = LRUCache(capacity=24)
        self.mask_cache = LRUCache(capacity=48)

        # SAM embedding readiness state
        self.sam_ready_for_image_key: str | None = None

        # Canvas + Scroll
        self.canvas = UnifiedCanvas()
        self.canvas.on_modified = self.recompute_dirty
        self.canvas.on_need_sam_ready = self.ensure_sam_ready_for_current_image

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)

        # Left list
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(300)
        self.list_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.populate_list()
        self.list_widget.currentRowChanged.connect(self.on_list_row_changed)

        # Mode buttons
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

        # Actions
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

        # Layout
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

        # Shortcuts
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_if_dirty_force)

        # Initial load
        self.list_widget.setCurrentRow(0)

    # ---------- Mode handlers ----------
    def on_click_brush(self):
        self.canvas.set_mode("brush")
        self.set_status("Brush mode")

    def on_click_erase(self):
        self.canvas.set_mode("erase")
        self.set_status("Eraser mode")

    def on_click_sam(self):
        # SAM 모드로 바꾸는 시점에 embedding을 준비 (딜레이 로딩)
        self.canvas.set_mode("sam")
        ok = self.ensure_sam_ready_for_current_image()
        if ok:
            self.set_status("SAM mode (ready)")
        else:
            # 준비 실패면 모드 유지/전환은 선택사항인데,
            # 여기서는 UX 상 brush로 돌려놓음
            self.btn_brush.setChecked(True)
            self.canvas.set_mode("brush")

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

    # ---------- Status / Cursor helpers ----------
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
            # discard
            self.set_baseline_from_current()
            self.set_status("Discarded changes")
            return True

    def closeEvent(self, event):
        if self.check_unsaved_changes():
            event.accept()
        else:
            event.ignore()

    # ---------- UI list ----------
    def populate_list(self):
        self.list_widget.clear()
        for p in self.mask_files:
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.list_widget.addItem(item)

    def on_list_row_changed(self, row: int):
        if row < 0 or row >= self.list_widget.count():
            return
        if row == self.index and self.current_mask_path is not None:
            return

        if not self.check_unsaved_changes():
            self.list_widget.blockSignals(True)
            self.list_widget.setCurrentRow(self.index if self.index >= 0 else 0)
            self.list_widget.blockSignals(False)
            return

        item = self.list_widget.item(row)
        path = Path(item.data(Qt.UserRole))
        self.index = row
        self.load_mask_path(path)

    def open_file_dialog(self):
        start_dir = str(MASK_DIR)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open mask PNG",
            start_dir,
            "PNG Images (*.png)"
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

        self.load_mask_path(p)
        self.set_status(f"Opened (external): {p.name}")

    # ---------- Loading ----------
    def find_image_path(self, image_key: str) -> Path | None:
        for ext in ("jpg", "png", "jpeg"):
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

    def load_mask_path(self, mask_path: Path):
        if not mask_path.exists():
            QMessageBox.warning(self, "Missing file", f"Mask not found:\n{mask_path}")
            return

        stem = mask_path.stem
        instance_id = parse_instance_id_from_stem(stem)
        image_key = instance_id.split("_obj")[0]

        image_path = self.find_image_path(image_key)
        if image_path is None:
            QMessageBox.warning(self, "Missing image", f"Image not found for:\n{image_key}\nin\n{IMAGE_DIR}")
            return

        img_bgr = self._read_image_cached(image_path)
        if img_bgr is None:
            QMessageBox.warning(self, "Read fail", f"Failed to read image:\n{image_path}")
            return

        mask = self._read_mask_cached(mask_path)
        if mask is None:
            QMessageBox.warning(self, "Read fail", f"Failed to read mask:\n{mask_path}")
            return

        self.current_mask_path = mask_path
        self.current_stem = stem
        self.current_image_key = image_key

        # IMPORTANT: 파일 이동 시 SAM embedding은 준비하지 않음 (전환 속도 우선)
        # 필요해지면(SAM 버튼/클릭) ensure_sam_ready_for_current_image에서 준비
        self.canvas.set_data(img_bgr, mask)

        # baseline reset
        self.set_baseline_from_current()
        self.set_status(f"Loaded: {mask_path.name}")

        QTimer.singleShot(0, self.focus_on_mask)

    # ---------- SAM lazy embedding ----------
    def ensure_sam_ready_for_current_image(self) -> bool:
        """
        현재 로드된 원본 이미지 기준으로 predictor.set_image()가 되어있지 않으면 수행.
        수행 중에는 로딩 커서(WaitCursor)로 표시.
        """
        if self.current_image_key is None:
            return False
        if self.sam_ready_for_image_key == self.current_image_key:
            return True

        # 이미지가 canvas에 이미 세팅되어 있으니 canvas.image_bgr를 사용
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
        if self.current_mask_path is None or self.current_stem is None:
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
        instance_id = parse_instance_id_from_stem(stem)
        image_id = instance_id.split("_obj")[0]

        save_mask_path = AFTER_MASK_DIR / f"{stem}_edited.png"
        ok = cv2.imwrite(str(save_mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            QMessageBox.warning(self, "Save failed", f"Failed to write:\n{save_mask_path}")
            return

        src_label_path = find_src_label_path(image_id)
        src_lines = read_label_lines(src_label_path)
        class_map = build_class_map_from_src_lines(src_lines)
        class_id = get_class_id_for_instance(class_map, instance_id)

        if class_id is None:
            QMessageBox.warning(
                self,
                "Class id missing",
                f"class_id not found for instance_id={instance_id}\nfrom\n{src_label_path}"
            )
            return

        bbox, poly = mask_to_bbox_and_polygon(mask)
        if bbox is None or poly is None or len(poly) < 3:
            QMessageBox.warning(self, "Invalid mask", f"Empty/invalid mask after edit:\n{instance_id}")
            return

        x1, y1, x2, y2 = bbox
        poly_flat = []
        for (px, py) in poly:
            poly_flat.append(f"{px:.2f}")
            poly_flat.append(f"{py:.2f}")

        new_line = (
            f"{instance_id} {class_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            + " ".join(poly_flat)
        )

        dst_label_path = AFTER_LABEL_DIR / f"{image_id}_edited.txt"
        dst_lines = read_label_lines(dst_label_path) if dst_label_path.exists() else [HEADER]
        dst_lines = upsert_instance_line(dst_lines, instance_id, new_line)
        dst_label_path.write_text("\n".join(dst_lines) + "\n", encoding="utf-8")

        # update baseline after save
        self.set_baseline_from_current()
        self.set_status(f"✅ Saved: {save_mask_path.name}")

    # ---------- Navigation ----------
    def next_image(self):
        if self.index < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(self.index + 1)
        else:
            self.close()

    def prev_image(self):
        if self.index > 0:
            self.list_widget.setCurrentRow(self.index - 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # (선택) 폰트 경고/플랫폼 폰트 불안정 방지
    # from PySide6.QtGui import QFont
    # app.setFont(QFont("Segoe UI", 10))

    win = UnifiedEditor()
    win.showMaximized()
    sys.exit(app.exec())