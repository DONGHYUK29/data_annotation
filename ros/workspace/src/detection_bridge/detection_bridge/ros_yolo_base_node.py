import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from detection_bridge_msgs.msg import Detection, DetectionArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from ultralytics import YOLO


class RosYoloBaseNode(Node):
    def __init__(self, node_name: str, default_use_depth_refine: bool, window_name: str):
        super().__init__(node_name)

        self.window_name = window_name
        self.default_use_depth_refine = default_use_depth_refine

        self.declare_parameter(
            "model_path",
            "",
        )
        self.declare_parameter("inference_device", "")
        self.declare_parameter("conf_thres", 0.25 if default_use_depth_refine else 0.50)
        self.declare_parameter("final_conf_thres", 0.25 if default_use_depth_refine else 0.50)
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_every_sec", 0.0)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("result_topic", "/detection_result")
        self.declare_parameter("result_json_topic", "/detection_result_json")
        self.declare_parameter("instance_mask_topic", "/instance_mask/image")
        self.declare_parameter("annotated_image_topic", "/annotated_image")
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("image_qos_reliability", "reliable")
        self.declare_parameter("debug_stamp_delta", False)
        self.declare_parameter("use_depth_refine", default_use_depth_refine)
        self.declare_parameter("use_refined_mask_center", True)
        self.declare_parameter("publish_json_debug", False)
        self.declare_parameter("publish_annotated_image", False)
        self.declare_parameter("publish_instance_mask", False)

        self.declare_parameter("depth_valid_min_m", 0.10)
        self.declare_parameter("depth_valid_max_m", 3.00)
        self.declare_parameter("depth_refine_abs_delta_m", 0.06)
        self.declare_parameter("depth_refine_rel_delta", 0.08)
        self.declare_parameter("depth_refine_erode_kernel", 5)
        self.declare_parameter("depth_refine_boundary_iter", 2)
        self.declare_parameter("depth_refine_close_kernel", 3)
        self.declare_parameter("depth_refine_bottom_keep_ratio", 0.30)
        self.declare_parameter("depth_refine_min_area_ratio", 0.85)
        self.declare_parameter("depth_refine_min_mask_area", 80)
        self.declare_parameter("depth_refine_min_core_pixels", 30)
        self.declare_parameter("depth_refine_min_valid_ratio", 0.20)
        self.declare_parameter("depth_refine_keep_largest_component", True)
        self.declare_parameter("draw_removed_pixels", True)
        self.declare_parameter("draw_original_contour", True)
        self.declare_parameter("draw_refined_contour", True)
        self.declare_parameter("mask_alpha", 0.35)
        self.declare_parameter("use_depth_score", False)
        self.declare_parameter("depth_quality_std_tau_m", 0.05)

        self.model_path = str(self.get_parameter("model_path").value).strip()
        if not self.model_path:
            self.model_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
        if not self.model_path:
            raise RuntimeError(
                "model_path is empty. Set the ROS parameter or YOLO_MODEL_PATH to a YOLO segmentation .pt file."
            )
        self.inference_device = str(self.get_parameter("inference_device").value).strip()
        if not self.inference_device:
            self.inference_device = "0" if torch.cuda.is_available() else "cpu"
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.final_conf_thres = float(self.get_parameter("final_conf_thres").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.publish_every_sec = float(self.get_parameter("publish_every_sec").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.result_topic = str(self.get_parameter("result_topic").value)
        self.result_json_topic = str(self.get_parameter("result_json_topic").value)
        self.instance_mask_topic = str(self.get_parameter("instance_mask_topic").value)
        self.annotated_image_topic = str(self.get_parameter("annotated_image_topic").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.image_qos_reliability = str(
            self.get_parameter("image_qos_reliability").value
        ).strip().lower()
        self.debug_stamp_delta = bool(self.get_parameter("debug_stamp_delta").value)
        self.use_depth_refine = bool(self.get_parameter("use_depth_refine").value)
        self.use_refined_mask_center = bool(
            self.get_parameter("use_refined_mask_center").value
        )
        self.publish_json_debug = bool(self.get_parameter("publish_json_debug").value)
        self.publish_annotated_image = bool(
            self.get_parameter("publish_annotated_image").value
        )
        self.publish_instance_mask = bool(self.get_parameter("publish_instance_mask").value)

        self.depth_valid_min_m = float(self.get_parameter("depth_valid_min_m").value)
        self.depth_valid_max_m = float(self.get_parameter("depth_valid_max_m").value)
        self.depth_refine_abs_delta_m = float(
            self.get_parameter("depth_refine_abs_delta_m").value
        )
        self.depth_refine_rel_delta = float(
            self.get_parameter("depth_refine_rel_delta").value
        )
        self.depth_refine_erode_kernel = int(
            self.get_parameter("depth_refine_erode_kernel").value
        )
        self.depth_refine_boundary_iter = int(
            self.get_parameter("depth_refine_boundary_iter").value
        )
        self.depth_refine_close_kernel = int(
            self.get_parameter("depth_refine_close_kernel").value
        )
        self.depth_refine_bottom_keep_ratio = float(
            self.get_parameter("depth_refine_bottom_keep_ratio").value
        )
        self.depth_refine_min_area_ratio = float(
            self.get_parameter("depth_refine_min_area_ratio").value
        )
        self.depth_refine_min_mask_area = int(
            self.get_parameter("depth_refine_min_mask_area").value
        )
        self.depth_refine_min_core_pixels = int(
            self.get_parameter("depth_refine_min_core_pixels").value
        )
        self.depth_refine_min_valid_ratio = float(
            self.get_parameter("depth_refine_min_valid_ratio").value
        )
        self.depth_refine_keep_largest_component = bool(
            self.get_parameter("depth_refine_keep_largest_component").value
        )
        self.draw_removed_pixels = bool(self.get_parameter("draw_removed_pixels").value)
        self.draw_original_contour = bool(
            self.get_parameter("draw_original_contour").value
        )
        self.draw_refined_contour = bool(
            self.get_parameter("draw_refined_contour").value
        )
        self.mask_alpha = float(self.get_parameter("mask_alpha").value)
        self.use_depth_score = bool(self.get_parameter("use_depth_score").value)
        self.depth_quality_std_tau_m = float(
            self.get_parameter("depth_quality_std_tau_m").value
        )

        self.depth_refine_erode_kernel = self._make_odd(self.depth_refine_erode_kernel)
        self.depth_refine_close_kernel = self._make_odd(self.depth_refine_close_kernel)
        self.depth_refine_boundary_iter = max(1, self.depth_refine_boundary_iter)
        self.depth_refine_bottom_keep_ratio = float(
            np.clip(self.depth_refine_bottom_keep_ratio, 0.0, 0.9)
        )
        self.depth_refine_min_area_ratio = float(
            np.clip(self.depth_refine_min_area_ratio, 0.0, 1.0)
        )

        self.result_pub = self.create_publisher(DetectionArray, self.result_topic, 10)
        self.json_pub = self.create_publisher(String, self.result_json_topic, 10)
        self.mask_pub = self.create_publisher(Image, self.instance_mask_topic, 10)
        self.annotated_pub = self.create_publisher(Image, self.annotated_image_topic, 10)

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO inference_device: {self.inference_device}")

        self.bridge = CvBridge()
        self.intr = None
        self.last_intrinsics_warning_time = 0.0
        image_qos = self.make_image_qos()
        self.color_sub = Subscriber(
            self, Image, self.color_topic, qos_profile=image_qos
        )
        self.depth_sub = Subscriber(
            self, Image, self.depth_topic, qos_profile=image_qos
        )
        self.color_sub.registerCallback(self.color_raw_callback)
        self.depth_sub.registerCallback(self.depth_raw_callback)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            image_qos,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=max(1, self.sync_queue_size),
            slop=max(0.0, self.sync_slop_sec),
        )
        self.sync.registerCallback(self.image_callback)

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 960)

        self.rgb_input_count = 0
        self.depth_input_count = 0
        self.input_frame_count = 0
        self.processed_frame_count = 0
        self.output_count = 0
        self.fps_window_start = time.perf_counter()
        self.rgb_input_fps = 0.0
        self.depth_input_fps = 0.0
        self.input_fps = 0.0
        self.processed_fps = 0.0
        self.output_fps = 0.0
        self.processing_fps = 0.0
        self.processing_latency_ms = 0.0
        self.preprocess_latency_ms = 0.0
        self.inference_latency_ms = 0.0
        self.postprocess_latency_ms = 0.0
        self.ros_message_latency_ms = 0.0
        self.end_to_end_latency_ms = 0.0
        self.publish_return_latency_ms = 0.0
        self.latency_equivalent_fps = 0.0
        self.last_publish_duration_ms = 0.0
        self.last_stamp_delta_ms = 0.0
        self.last_stamp_debug_time = 0.0
        self.last_publish_time = 0.0
        self.performance_timer = self.create_timer(1.0, self.update_window_fps)

        self.get_logger().info(f"ROS YOLO node started: {node_name}")
        self.get_logger().info(f"color_topic: {self.color_topic}")
        self.get_logger().info(f"depth_topic: {self.depth_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"result_topic: {self.result_topic}")
        self.get_logger().info(f"instance_mask_topic: {self.instance_mask_topic}")
        self.get_logger().info(f"annotated_image_topic: {self.annotated_image_topic}")
        self.get_logger().info(f"use_depth_refine: {self.use_depth_refine}")
        self.get_logger().info(
            "sync: RGB + Depth only, CameraInfo uses latest value, "
            f"queue={self.sync_queue_size}, slop={self.sync_slop_sec:.3f}s, "
            f"qos={self.image_qos_reliability}"
        )

    def color_raw_callback(self, _msg: Image) -> None:
        self.rgb_input_count += 1

    def depth_raw_callback(self, _msg: Image) -> None:
        self.depth_input_count += 1

    def camera_info_callback(self, camera_info_msg: CameraInfo) -> None:
        self.update_intrinsics(camera_info_msg)

    def make_image_qos(self) -> QoSProfile:
        if self.image_qos_reliability not in {"reliable", "best_effort"}:
            raise ValueError(
                "image_qos_reliability must be 'reliable' or 'best_effort', got "
                f"{self.image_qos_reliability!r}"
            )
        reliability = (
            QoSReliabilityPolicy.RELIABLE
            if self.image_qos_reliability == "reliable"
            else QoSReliabilityPolicy.BEST_EFFORT
        )
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=max(1, self.sync_queue_size),
            reliability=reliability,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

    def image_callback(self, color_msg: Image, depth_msg: Image):
        t_recv = time.perf_counter()
        self.input_frame_count += 1
        try:
            if self.intr is None:
                now = time.perf_counter()
                if now - self.last_intrinsics_warning_time >= 5.0:
                    self.get_logger().warning(
                        "Skipping synchronized RGB-D frame until CameraInfo arrives on "
                        f"{self.camera_info_topic}"
                    )
                    self.last_intrinsics_warning_time = now
                return

            self.last_stamp_delta_ms = abs(
                self.stamp_to_sec(color_msg) - self.stamp_to_sec(depth_msg)
            ) * 1000.0
            if (
                self.debug_stamp_delta
                and t_recv - self.last_stamp_debug_time >= 1.0
            ):
                self.get_logger().debug(
                    "RGB/Depth stamp delta: "
                    f"{self.last_stamp_delta_ms:.3f} ms "
                    f"(slop={self.sync_slop_sec * 1000.0:.1f} ms)"
                )
                self.last_stamp_debug_time = t_recv
            needs_visual = self.show_window or self.publish_annotated_image

            preprocess_start = time.perf_counter()
            frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_m = self.depth_msg_to_meters(depth_msg)
            preprocess_end = time.perf_counter()
            instance_mask = (
                np.zeros(frame.shape[:2], dtype=np.uint16)
                if self.publish_instance_mask
                else None
            )

            inference_start = preprocess_end
            results = self.model(
                frame,
                conf=self.conf_thres,
                verbose=False,
                retina_masks=True,
                device=self.inference_device,
            )
            inference_end = time.perf_counter()

            detections: List[Detection] = []
            objects: List[Dict[str, Any]] = []
            visual_entries: List[Dict[str, Any]] = []
            instance_id = 1

            for r in results:
                if r.boxes is None:
                    continue

                boxes = r.boxes
                masks_np = self.extract_masks_from_result(r, frame.shape[:2])

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = str(self.model.names[class_id])

                    x1, y1, x2, y2 = self.clamp_bbox(
                        x1, y1, x2, y2, frame.shape[1], frame.shape[0]
                    )

                    if masks_np is not None and i < len(masks_np):
                        original_mask = (masks_np[i] > 0.5).astype(np.uint8)
                    else:
                        original_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        original_mask[y1 : y2 + 1, x1 : x2 + 1] = 1

                    refined_mask = original_mask.copy()
                    removed_mask = np.zeros_like(original_mask, dtype=np.uint8)
                    refine_stats: Dict[str, Any] = {
                        "enabled": bool(self.use_depth_refine),
                        "applied": False,
                        "reason": "disabled" if not self.use_depth_refine else "not_run",
                        "removed_pixels": 0,
                        "area_ratio": 1.0,
                    }

                    if self.use_depth_refine and original_mask.sum() > 0:
                        refined_mask, removed_mask, refine_stats = self.refine_mask_with_depth(
                            original_mask, depth_m, (x1, y1, x2, y2)
                        )

                    depth_quality = self.compute_depth_quality(refined_mask, depth_m)
                    final_score = conf
                    if self.use_depth_score:
                        final_score = conf * (0.5 + 0.5 * depth_quality)

                    if final_score < self.final_conf_thres:
                        continue

                    bbox_center = self.get_bbox_center((x1, y1, x2, y2))
                    mask_center = self.get_object_center(
                        refined_mask,
                        (x1, y1, x2, y2),
                        use_mask=True,
                    )
                    cx, cy = mask_center if self.use_refined_mask_center else bbox_center
                    depth = self.get_robust_depth_from_mask(depth_m, refined_mask)
                    if depth == 0:
                        depth = self.get_valid_depth_near_center_np(depth_m, cx, cy)
                    depth_valid = depth > 0.0
                    if depth_valid:
                        x = (cx - self.intr["ppx"]) / self.intr["fx"] * depth
                        y = (cy - self.intr["ppy"]) / self.intr["fy"] * depth
                        z = depth
                    else:
                        x, y, z = 0.0, 0.0, 0.0

                    if instance_mask is not None:
                        mask_bool = refined_mask > 0
                        instance_mask[mask_bool] = np.uint16(min(instance_id, 65535))

                    det = Detection()
                    det.instance_id = int(instance_id)
                    det.class_id = int(class_id)
                    det.class_name = class_name
                    det.confidence = float(conf)
                    det.final_confidence = float(final_score)
                    det.depth_quality = float(depth_quality)
                    det.bbox_xyxy = [int(x1), int(y1), int(x2), int(y2)]
                    det.center_xy = [int(cx), int(cy)]
                    det.bbox_center_xy = [int(bbox_center[0]), int(bbox_center[1])]
                    det.mask_center_xy = [int(mask_center[0]), int(mask_center[1])]
                    det.depth_valid = bool(depth_valid)
                    det.depth_m = float(depth)
                    det.xyz_m = [float(x), float(y), float(z)]
                    det.mask_area = int(original_mask.sum())
                    det.refined_mask_area = int(refined_mask.sum())
                    det.depth_refine_applied = bool(refine_stats.get("applied", False))
                    det.depth_refine_reason = str(refine_stats.get("reason", ""))
                    det.depth_refine_removed_pixels = int(
                        refine_stats.get("removed_pixels", 0)
                    )
                    det.depth_refine_area_ratio = float(
                        refine_stats.get("area_ratio", 1.0)
                    )
                    detections.append(det)

                    objects.append(
                        {
                            "instance_id": int(instance_id),
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "confidence": float(conf),
                            "final_confidence": float(final_score),
                            "depth_quality": float(depth_quality),
                            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                            "center_xy": [int(cx), int(cy)],
                            "bbox_center_xy": [int(bbox_center[0]), int(bbox_center[1])],
                            "mask_center_xy": [int(mask_center[0]), int(mask_center[1])],
                            "depth_valid": bool(depth_valid),
                            "depth_m": float(depth),
                            "xyz_m": [float(x), float(y), float(z)],
                            "mask_area": int(original_mask.sum()),
                            "refined_mask_area": int(refined_mask.sum()),
                            "depth_refine": refine_stats,
                        }
                    )

                    if needs_visual:
                        visual_entries.append(
                            {
                                "original_mask": original_mask,
                                "refined_mask": refined_mask,
                                "removed_mask": removed_mask,
                                "bbox": (x1, y1, x2, y2),
                                "center": (cx, cy),
                                "class_name": class_name,
                                "instance_id": instance_id,
                                "conf": conf,
                                "final_score": final_score,
                                "xyz": (x, y, z),
                                "refine_stats": refine_stats,
                            }
                        )
                    instance_id += 1

            postprocess_end = time.perf_counter()
            self.processed_frame_count += 1

            now = postprocess_end
            should_publish = self.publish_every_sec <= 0.0 or (
                now - self.last_publish_time >= self.publish_every_sec
            )

            metrics_elapsed = max(1e-9, time.perf_counter() - self.fps_window_start)
            rgb_input_fps_snapshot = self.rgb_input_count / metrics_elapsed
            depth_input_fps_snapshot = self.depth_input_count / metrics_elapsed
            input_fps_snapshot = self.input_frame_count / metrics_elapsed
            processed_fps_snapshot = self.processed_frame_count / metrics_elapsed
            output_fps_snapshot = (
                self.output_count + (1 if should_publish else 0)
            ) / metrics_elapsed

            ros_message_start = time.perf_counter()
            result_msg = DetectionArray()
            result_msg.header = color_msg.header
            result_msg.input_fps = float(input_fps_snapshot)
            result_msg.processed_fps = float(processed_fps_snapshot)
            result_msg.output_fps = float(output_fps_snapshot)
            result_msg.depth_refine_enabled = bool(self.use_depth_refine)
            result_msg.detections = detections
            result_msg.rgb_input_fps = float(rgb_input_fps_snapshot)
            result_msg.depth_input_fps = float(depth_input_fps_snapshot)
            result_msg.synced_input_fps = float(input_fps_snapshot)
            result_msg.publish_fps = float(output_fps_snapshot)

            mask_msg = None
            if self.publish_instance_mask and instance_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(instance_mask, encoding="16UC1")
                mask_msg.header = color_msg.header

            t_publish_ready = time.perf_counter()
            self.preprocess_latency_ms = (preprocess_end - preprocess_start) * 1000.0
            self.inference_latency_ms = (inference_end - inference_start) * 1000.0
            self.postprocess_latency_ms = (ros_message_start - inference_end) * 1000.0
            self.ros_message_latency_ms = (t_publish_ready - ros_message_start) * 1000.0
            self.processing_latency_ms = (t_publish_ready - t_recv) * 1000.0
            self.processing_fps = 1.0 / max(1e-9, t_publish_ready - t_recv)
            self.end_to_end_latency_ms = float(self.processing_latency_ms)
            self.latency_equivalent_fps = float(self.processing_fps)

            result_msg.processing_fps = float(self.processing_fps)
            result_msg.processing_latency_ms = float(self.processing_latency_ms)
            result_msg.end_to_end_latency_ms = float(self.end_to_end_latency_ms)
            result_msg.latency_equivalent_fps = float(self.latency_equivalent_fps)
            result_msg.preprocess_latency_ms = float(self.preprocess_latency_ms)
            result_msg.inference_latency_ms = float(self.inference_latency_ms)
            result_msg.postprocess_latency_ms = float(self.postprocess_latency_ms)
            result_msg.ros_message_latency_ms = float(self.ros_message_latency_ms)

            if should_publish:
                self.result_pub.publish(result_msg)
                t_publish_return = time.perf_counter()
                self.last_publish_duration_ms = (
                    t_publish_return - t_publish_ready
                ) * 1000.0
                self.publish_return_latency_ms = (
                    t_publish_return - t_recv
                ) * 1000.0

                if self.publish_instance_mask and mask_msg is not None:
                    self.mask_pub.publish(mask_msg)

                if self.publish_json_debug:
                    payload = {
                        "input_fps": float(input_fps_snapshot),
                        "processed_fps": float(processed_fps_snapshot),
                        "processing_fps": float(self.processing_fps),
                        "processing_latency_ms": float(self.processing_latency_ms),
                        "output_fps": float(output_fps_snapshot),
                        "publish_fps": float(output_fps_snapshot),
                        "rgb_input_fps": float(rgb_input_fps_snapshot),
                        "depth_input_fps": float(depth_input_fps_snapshot),
                        "synced_input_fps": float(input_fps_snapshot),
                        "end_to_end_latency_ms": float(self.end_to_end_latency_ms),
                        "publish_return_latency_ms": float(
                            self.publish_return_latency_ms
                        ),
                        "publish_call_latency_ms": float(
                            self.last_publish_duration_ms
                        ),
                        "latency_equivalent_fps": float(self.latency_equivalent_fps),
                        "preprocess_latency_ms": float(self.preprocess_latency_ms),
                        "inference_latency_ms": float(self.inference_latency_ms),
                        "postprocess_latency_ms": float(self.postprocess_latency_ms),
                        "ros_message_latency_ms": float(self.ros_message_latency_ms),
                        "color_stamp_sec": self.stamp_to_sec(color_msg),
                        "depth_stamp_sec": self.stamp_to_sec(depth_msg),
                        "stamp_delta_ms": float(self.last_stamp_delta_ms),
                        "depth_refine_enabled": bool(self.use_depth_refine),
                        "objects": objects,
                    }
                    json_msg = String()
                    json_msg.data = json.dumps(payload, ensure_ascii=False)
                    self.json_pub.publish(json_msg)

                self.output_count += 1
                self.last_publish_time = t_publish_return

            if needs_visual:
                visual_start = time.perf_counter()
                annotated = frame.copy()
                for entry in visual_entries:
                    self.draw_object(
                        annotated=annotated,
                        original_mask=entry["original_mask"],
                        refined_mask=entry["refined_mask"],
                        removed_mask=entry["removed_mask"],
                        bbox=entry["bbox"],
                        center=entry["center"],
                        class_name=entry["class_name"],
                        instance_id=entry["instance_id"],
                        conf=entry["conf"],
                        final_score=entry["final_score"],
                        xyz=entry["xyz"],
                        refine_stats=entry["refine_stats"],
                    )
                self.draw_fps_overlay(annotated)

                if self.publish_annotated_image and should_publish:
                    annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                    annotated_msg.header = color_msg.header
                    self.annotated_pub.publish(annotated_msg)

                visual_end = time.perf_counter()
                visual_latency_ms = (visual_end - visual_start) * 1000.0
            else:
                annotated = None
                visual_latency_ms = 0.0

            if self.show_window and annotated is not None:
                cv2.imshow(self.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.get_logger().info("ESC pressed. Closing node...")
                    raise KeyboardInterrupt

        except RuntimeError as e:
            self.get_logger().error(f"RuntimeError in camera/inference loop: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

    def update_window_fps(self) -> None:
        elapsed = time.perf_counter() - self.fps_window_start
        if elapsed <= 0.0:
            return
        self.rgb_input_fps = self.rgb_input_count / elapsed
        self.depth_input_fps = self.depth_input_count / elapsed
        self.input_fps = self.input_frame_count / elapsed
        self.processed_fps = self.processed_frame_count / elapsed
        self.output_fps = self.output_count / elapsed
        if (
            self.rgb_input_count
            or self.depth_input_count
            or self.input_frame_count
            or self.processed_frame_count
            or self.output_count
        ):
            self.get_logger().info(
                "PERF rgb={:.2f} depth={:.2f} synced={:.2f} "
                "processed={:.2f} publish={:.2f} latency_ready={:.2f}ms "
                "latency_return={:.2f}ms eq={:.2f}fps pre={:.2f}ms "
                "infer={:.2f}ms post={:.2f}ms msg={:.2f}ms "
                "pub_call={:.2f}ms stamp_delta={:.2f}ms".format(
                    self.rgb_input_fps,
                    self.depth_input_fps,
                    self.input_fps,
                    self.processed_fps,
                    self.output_fps,
                    self.end_to_end_latency_ms,
                    self.publish_return_latency_ms,
                    self.latency_equivalent_fps,
                    self.preprocess_latency_ms,
                    self.inference_latency_ms,
                    self.postprocess_latency_ms,
                    self.ros_message_latency_ms,
                    self.last_publish_duration_ms,
                    self.last_stamp_delta_ms,
                )
            )
        self.rgb_input_count = 0
        self.depth_input_count = 0
        self.input_frame_count = 0
        self.processed_frame_count = 0
        self.output_count = 0
        self.fps_window_start = time.perf_counter()

    def update_intrinsics(self, camera_info_msg: CameraInfo) -> None:
        self.intr = {
            "fx": float(camera_info_msg.k[0]),
            "fy": float(camera_info_msg.k[4]),
            "ppx": float(camera_info_msg.k[2]),
            "ppy": float(camera_info_msg.k[5]),
        }

    def depth_msg_to_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth_msg.encoding == "16UC1":
            return depth.astype(np.float32) * self.depth_scale
        if depth_msg.encoding == "32FC1":
            return depth.astype(np.float32)
        return depth.astype(np.float32) * self.depth_scale

    @staticmethod
    def stamp_to_sec(msg) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def extract_masks_from_result(
        self, result, hw: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        if result.masks is None:
            return None
        h, w = hw
        try:
            masks = result.masks.data.cpu().numpy()
        except Exception:
            return None
        if masks.ndim != 3:
            return None
        resized = []
        for m in masks:
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(m.astype(np.float32))
        return np.stack(resized, axis=0) if len(resized) > 0 else None

    @staticmethod
    def clamp_bbox(
        x1: int, y1: int, x2: int, y2: int, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def refine_mask_with_depth(
        self, mask: np.ndarray, depth_m: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        h_img, w_img = mask.shape[:2]
        x1, y1, x2, y2 = self.clamp_bbox(*bbox, width=w_img, height=h_img)
        stats: Dict[str, Any] = {
            "enabled": True,
            "applied": False,
            "reason": "init",
            "z_obj_m": 0.0,
            "delta_m": 0.0,
            "removed_pixels": 0,
            "area_ratio": 1.0,
        }

        roi_mask = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
        roi_depth = depth_m[y1 : y2 + 1, x1 : x2 + 1]
        original_area = int(roi_mask.sum())
        if original_area < self.depth_refine_min_mask_area:
            stats["reason"] = "small_mask"
            return mask.copy(), np.zeros_like(mask, dtype=np.uint8), stats

        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.depth_refine_erode_kernel, self.depth_refine_erode_kernel),
        )
        core = cv2.erode(roi_mask, erode_kernel, iterations=1)
        if int(core.sum()) < self.depth_refine_min_core_pixels:
            core = roi_mask.copy()

        valid_depth = self.get_valid_depth_mask(roi_depth)
        core_valid = (core > 0) & valid_depth
        valid_count = int(core_valid.sum())
        if valid_count < self.depth_refine_min_core_pixels:
            stats["reason"] = "not_enough_core_depth"
            return mask.copy(), np.zeros_like(mask, dtype=np.uint8), stats

        valid_ratio = valid_count / max(1, int(core.sum()))
        if valid_ratio < self.depth_refine_min_valid_ratio:
            stats["reason"] = "low_valid_depth_ratio"
            stats["valid_ratio"] = float(valid_ratio)
            return mask.copy(), np.zeros_like(mask, dtype=np.uint8), stats

        z_obj = float(np.median(roi_depth[core_valid]))
        delta = max(self.depth_refine_abs_delta_m, z_obj * self.depth_refine_rel_delta)
        stats["z_obj_m"] = z_obj
        stats["delta_m"] = float(delta)
        stats["valid_ratio"] = float(valid_ratio)

        eroded_for_boundary = cv2.erode(
            roi_mask, erode_kernel, iterations=self.depth_refine_boundary_iter
        )
        boundary = (roi_mask > 0) & (eroded_for_boundary == 0)
        roi_h, roi_w = roi_mask.shape[:2]
        if roi_h <= 0 or roi_w <= 0:
            stats["reason"] = "empty_roi"
            return mask.copy(), np.zeros_like(mask, dtype=np.uint8), stats

        keep_bottom_start_y = int(round(roi_h * (1.0 - self.depth_refine_bottom_keep_ratio)))
        yy = np.arange(roi_h).reshape(-1, 1)
        not_bottom_zone = yy < keep_bottom_start_y
        depth_diff = np.abs(roi_depth - z_obj)
        eligible = boundary & not_bottom_zone & valid_depth
        remove_roi = eligible & (depth_diff > delta)

        refined_roi = roi_mask.copy()
        refined_roi[remove_roi] = 0
        if self.depth_refine_close_kernel >= 3:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.depth_refine_close_kernel, self.depth_refine_close_kernel),
            )
            refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_CLOSE, close_kernel)
        if self.depth_refine_keep_largest_component:
            refined_roi = self.keep_largest_component(refined_roi)

        refined_area = int(refined_roi.sum())
        area_ratio = refined_area / max(1, original_area)
        removed_pixels = int(original_area - refined_area)
        stats["removed_pixels"] = int(max(0, removed_pixels))
        stats["area_ratio"] = float(area_ratio)
        if refined_area <= 0:
            stats["reason"] = "empty_after_refine_reverted"
            return mask.copy(), np.zeros_like(mask, dtype=np.uint8), stats
        if area_ratio < self.depth_refine_min_area_ratio:
            stats["reason"] = "too_much_removed_reverted"
            return mask.copy(), np.zeros_like(mask, dtype=np.uint8), stats

        refined_full = mask.copy().astype(np.uint8)
        refined_full[y1 : y2 + 1, x1 : x2 + 1] = refined_roi.astype(np.uint8)
        removed_full = np.zeros_like(mask, dtype=np.uint8)
        removed_full[y1 : y2 + 1, x1 : x2 + 1] = (
            (roi_mask > 0) & (refined_roi == 0)
        ).astype(np.uint8)
        stats["applied"] = True
        stats["reason"] = "ok"
        return refined_full, removed_full, stats

    def get_valid_depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(depth_m)
            & (depth_m > self.depth_valid_min_m)
            & (depth_m < self.depth_valid_max_m)
        )

    @staticmethod
    def keep_largest_component(mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        if num_labels <= 1:
            return mask_u8
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas) + 1)
        return (labels == largest_label).astype(np.uint8)

    def compute_depth_quality(self, mask: np.ndarray, depth_m: np.ndarray) -> float:
        mask_bool = mask > 0
        area = int(mask_bool.sum())
        if area <= 0:
            return 0.0
        valid = mask_bool & self.get_valid_depth_mask(depth_m)
        valid_count = int(valid.sum())
        if valid_count <= 0:
            return 0.0
        valid_ratio = valid_count / max(1, area)
        vals = depth_m[valid]
        if vals.size <= 1:
            return float(np.clip(valid_ratio, 0.0, 1.0))
        depth_std = float(np.std(vals))
        q_var = float(np.exp(-depth_std / max(1e-6, self.depth_quality_std_tau_m)))
        quality = 0.6 * valid_ratio + 0.4 * q_var
        return float(np.clip(quality, 0.0, 1.0))

    def get_object_center(
        self, mask: np.ndarray, bbox: Tuple[int, int, int, int], use_mask: bool = True
    ) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        if use_mask and mask is not None and int(mask.sum()) > 0:
            m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
            if m["m00"] > 0:
                cx = int(round(m["m10"] / m["m00"]))
                cy = int(round(m["m01"] / m["m00"]))
                return cx, cy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_robust_depth_from_mask(self, depth_m: np.ndarray, mask: np.ndarray) -> float:
        if mask is None or int(mask.sum()) <= 0:
            return 0.0
        mask_u8 = mask.astype(np.uint8)
        kernel_size = max(3, self.depth_refine_erode_kernel)
        kernel_size = self._make_odd(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        core = cv2.erode(mask_u8, kernel, iterations=1)
        if int(core.sum()) < self.depth_refine_min_core_pixels:
            core = mask_u8
        valid = (core > 0) & self.get_valid_depth_mask(depth_m)
        if int(valid.sum()) <= 0:
            return 0.0
        return float(np.median(depth_m[valid]))

    def get_valid_depth_near_center_np(self, depth_m: np.ndarray, cx: int, cy: int) -> float:
        radius = 2
        h, w = depth_m.shape[:2]
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        roi = depth_m[y1:y2, x1:x2]
        valid = self.get_valid_depth_mask(roi)
        if int(valid.sum()) == 0:
            return 0.0
        return float(np.median(roi[valid]))

    def draw_object(
        self,
        annotated: np.ndarray,
        original_mask: np.ndarray,
        refined_mask: np.ndarray,
        removed_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        center: Tuple[int, int],
        class_name: str,
        instance_id: int,
        conf: float,
        final_score: float,
        xyz: Tuple[float, float, float],
        refine_stats: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = bbox
        cx, cy = center
        x, y, z = xyz

        if refined_mask is not None and int(refined_mask.sum()) > 0:
            mask_bool = refined_mask.astype(bool)
            green = np.zeros_like(annotated)
            green[:, :] = (0, 255, 0)
            annotated[mask_bool] = (
                annotated[mask_bool].astype(np.float32) * (1.0 - self.mask_alpha)
                + green[mask_bool].astype(np.float32) * self.mask_alpha
            ).astype(np.uint8)

        if self.draw_removed_pixels and removed_mask is not None and int(removed_mask.sum()) > 0:
            removed_bool = removed_mask.astype(bool)
            red = np.zeros_like(annotated)
            red[:, :] = (0, 0, 255)
            annotated[removed_bool] = (
                annotated[removed_bool].astype(np.float32) * 0.35
                + red[removed_bool].astype(np.float32) * 0.65
            ).astype(np.uint8)

        if self.draw_original_contour and original_mask is not None:
            contours, _ = cv2.findContours(
                original_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (255, 255, 255), 1)

        if self.draw_refined_contour and refined_mask is not None:
            contours, _ = cv2.findContours(
                refined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        depth_text = f"Z:{z:.2f}m" if z > 0.0 else "Z:N/A"
        if self.use_depth_score:
            label = (
                f"#{instance_id} {class_name} {conf:.2f}->{final_score:.2f} "
                f"{depth_text}"
            )
        else:
            label = f"#{instance_id} {class_name} {conf:.2f} {depth_text}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        coord_text = (
            f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
            if z > 0.0
            else "XYZ:N/A"
        )
        cv2.putText(
            annotated,
            coord_text,
            (x1, min(annotated.shape[0] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
        )
        if refine_stats.get("enabled", False):
            removed = int(refine_stats.get("removed_pixels", 0))
            area_ratio = float(refine_stats.get("area_ratio", 1.0))
            reason = str(refine_stats.get("reason", ""))
            ref_text = f"ref:{reason} rm:{removed} ar:{area_ratio:.2f}"
            cv2.putText(
                annotated,
                ref_text,
                (x1, min(annotated.shape[0] - 10, y2 + 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    def draw_fps_overlay(self, annotated: np.ndarray) -> None:
        lines = [
            (
                f"IN  R {self.rgb_input_fps:.1f}  D {self.depth_input_fps:.1f}  "
                f"S {self.input_fps:.1f}"
            ),
            (
                f"OUT proc {self.processed_fps:.1f}  pub {self.output_fps:.1f}  "
                f"depth {'ON' if self.use_depth_refine else 'OFF'}"
            ),
            (
                f"LAT {self.end_to_end_latency_ms:.1f}ms  "
                f"infer {self.inference_latency_ms:.1f}ms"
            ),
            (
                f"pre {self.preprocess_latency_ms:.1f}  "
                f"post {self.postprocess_latency_ms:.1f}  "
                f"msg {self.ros_message_latency_ms:.1f}  "
                f"pub {self.last_publish_duration_ms:.1f}ms"
            ),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        line_height = 15
        padding = 6
        panel_width = max(
            cv2.getTextSize(text, font, font_scale, thickness)[0][0]
            for text in lines
        ) + padding * 2
        panel_height = line_height * len(lines) + padding * 2
        x1 = max(0, annotated.shape[1] - panel_width - 6)
        y1 = 6
        x2 = min(annotated.shape[1], x1 + panel_width)
        y2 = min(annotated.shape[0], y1 + panel_height)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, annotated, 0.42, 0.0, annotated)

        for idx, text in enumerate(lines):
            cv2.putText(
                annotated,
                text,
                (x1 + padding, y1 + padding + 11 + idx * line_height),
                font,
                font_scale,
                (170, 255, 170) if idx < 2 else (180, 240, 255),
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _make_odd(value: int) -> int:
        value = max(1, int(value))
        if value % 2 == 0:
            value += 1
        return value

    def destroy_node(self):
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
                time.sleep(0.2)
            except Exception as e:
                self.get_logger().warning(f"cv2 cleanup failed: {e}")
        super().destroy_node()
