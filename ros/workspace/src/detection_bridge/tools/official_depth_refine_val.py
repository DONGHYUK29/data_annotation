#!/usr/bin/env python3
"""Ultralytics official segmentation validator with optional depth mask refine.

This uses Ultralytics' SegmentationValidator and metrics implementation. When
--use-depth-refine is disabled, results should match `yolo segment val` for the
same arguments. When enabled, only predicted masks are refined before the
official mask-IoU/AP computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator

KERI_ROOT = Path(__file__).resolve().parents[5]
EVALUATION_ROOT = KERI_ROOT / "sample_data" / "rgbd"


class DepthRefineSegmentationValidator(SegmentationValidator):
    def __init__(self, *args, depth_root=None, use_depth_refine=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_root = Path(depth_root).expanduser().resolve() if depth_root else None
        self.use_depth_refine = bool(use_depth_refine)
        self.depth_scale = 0.001
        self.depth_valid_min_m = 0.10
        self.depth_valid_max_m = 3.00
        self.abs_delta_m = 0.06
        self.rel_delta = 0.08
        self.erode_kernel = 5
        self.boundary_iter = 2
        self.close_kernel = 3
        self.bottom_keep_ratio = 0.30
        self.min_area_ratio = 0.85
        self.min_mask_area = 80
        self.min_core_pixels = 30
        self.min_valid_ratio = 0.20

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        if self.use_depth_refine and self.depth_root is not None and preds["masks"].shape[0] > 0:
            preds = dict(preds)
            preds["masks"] = self.refine_pred_masks(preds["masks"], batch)
        return super()._process_batch(preds, batch)

    def refine_pred_masks(self, masks: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        depth_m = self.load_depth_for_image(Path(batch["im_file"]))
        if depth_m is None:
            return masks

        mask_shape = tuple(int(x) for x in masks.shape[-2:])
        depth_for_mask = self.depth_to_mask_space(depth_m, batch, mask_shape)
        refined = []
        for mask_t in masks:
            mask = mask_t.detach().cpu().numpy().astype(np.uint8)
            refined_mask = self.refine_mask_with_depth(mask, depth_for_mask)
            refined.append(torch.from_numpy(refined_mask).to(device=masks.device, dtype=masks.dtype))
        return torch.stack(refined, dim=0) if refined else masks

    def load_depth_for_image(self, image_path: Path) -> Optional[np.ndarray]:
        if self.depth_root is None:
            return None
        depth_path = self.depth_root / f"{image_path.stem}.depth"
        if not depth_path.exists():
            return None
        try:
            depth = np.load(depth_path).astype(np.float32)
        except Exception:
            return None
        if depth.max(initial=0.0) > 20.0:
            depth = depth * self.depth_scale
        return depth

    def depth_to_mask_space(
        self,
        depth_m: np.ndarray,
        batch: dict[str, Any],
        mask_shape: Tuple[int, int],
    ) -> np.ndarray:
        # Official masks are evaluated in the same letterboxed image space as GT masks.
        target_h, target_w = mask_shape
        ori_h, ori_w = int(batch["ori_shape"][0]), int(batch["ori_shape"][1])
        if depth_m.shape[:2] != (ori_h, ori_w):
            depth_m = cv2.resize(depth_m, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

        ratio_pad = batch.get("ratio_pad", None)
        gain = min(target_h / max(ori_h, 1), target_w / max(ori_w, 1))
        pad_w = (target_w - ori_w * gain) / 2
        pad_h = (target_h - ori_h * gain) / 2
        if ratio_pad is not None:
            try:
                ratio = ratio_pad[0]
                pad = ratio_pad[1]
                if isinstance(ratio, (tuple, list)):
                    gain = float(ratio[0])
                else:
                    gain = float(ratio)
                pad_w = float(pad[0])
                pad_h = float(pad[1])
            except Exception:
                pass

        resized_w = max(1, int(round(ori_w * gain)))
        resized_h = max(1, int(round(ori_h * gain)))
        resized = cv2.resize(depth_m, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target_h, target_w), dtype=np.float32)
        left = int(round(pad_w))
        top = int(round(pad_h))
        right = min(target_w, left + resized_w)
        bottom = min(target_h, top + resized_h)
        src_w = max(0, right - left)
        src_h = max(0, bottom - top)
        if src_w > 0 and src_h > 0:
            canvas[top:bottom, left:right] = resized[:src_h, :src_w]
        return canvas

    def valid_depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(depth_m)
            & (depth_m > self.depth_valid_min_m)
            & (depth_m < self.depth_valid_max_m)
        )

    @staticmethod
    def largest_component(mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        if num_labels <= 1:
            return mask_u8
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        return (labels == largest).astype(np.uint8)

    def refine_mask_with_depth(self, mask: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        if int(mask.sum()) < self.min_mask_area:
            return mask

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return mask
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        roi_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        roi_depth = depth_m[y1 : y2 + 1, x1 : x2 + 1]
        original_area = int(roi_mask.sum())

        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.erode_kernel, self.erode_kernel)
        )
        core = cv2.erode(roi_mask, erode_kernel, iterations=1)
        if int(core.sum()) < self.min_core_pixels:
            core = roi_mask.copy()

        valid_depth = self.valid_depth_mask(roi_depth)
        core_valid = (core > 0) & valid_depth
        valid_count = int(core_valid.sum())
        if valid_count < self.min_core_pixels:
            return mask
        valid_ratio = valid_count / max(1, int(core.sum()))
        if valid_ratio < self.min_valid_ratio:
            return mask

        z_obj = float(np.median(roi_depth[core_valid]))
        delta = max(self.abs_delta_m, z_obj * self.rel_delta)
        eroded_boundary = cv2.erode(roi_mask, erode_kernel, iterations=self.boundary_iter)
        boundary = (roi_mask > 0) & (eroded_boundary == 0)
        keep_bottom_start = int(round(roi_mask.shape[0] * (1.0 - self.bottom_keep_ratio)))
        yy = np.arange(roi_mask.shape[0]).reshape(-1, 1)
        remove = boundary & (yy < keep_bottom_start) & valid_depth & (np.abs(roi_depth - z_obj) > delta)

        refined_roi = roi_mask.copy()
        refined_roi[remove] = 0
        if self.close_kernel >= 3:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel)
            )
            refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_CLOSE, close_kernel)
        refined_roi = self.largest_component(refined_roi)

        refined_area = int(refined_roi.sum())
        if refined_area <= 0:
            return mask
        if refined_area / max(1, original_area) < self.min_area_ratio:
            return mask

        refined = mask.copy()
        refined[y1 : y2 + 1, x1 : x2 + 1] = refined_roi
        return refined.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO segmentation .pt file")
    parser.add_argument("--data", default=str(EVALUATION_ROOT / "dataset.yaml"))
    parser.add_argument("--depth-root", default=str(EVALUATION_ROOT / "depth"))
    parser.add_argument("--use-depth-refine", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default=str(KERI_ROOT / "results" / "generated"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    name = args.name
    if not name:
        name = "dh_best_test_integrated_depth_refine_val" if args.use_depth_refine else "dh_best_test_integrated_official_subclass_val"

    model = YOLO(args.model)
    depth_root_arg = args.depth_root
    use_depth_refine_arg = args.use_depth_refine

    def validator_factory(args=None, _callbacks=None):
        return DepthRefineSegmentationValidator(
            args=args,
            _callbacks=_callbacks,
            depth_root=depth_root_arg,
            use_depth_refine=use_depth_refine_arg,
        )

    stats = model.val(
        validator=validator_factory,
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=name,
        exist_ok=True,
        plots=args.plots,
        save_json=args.save_json,
    )
    if hasattr(stats, "results_dict"):
        print(stats.results_dict)
    else:
        print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
