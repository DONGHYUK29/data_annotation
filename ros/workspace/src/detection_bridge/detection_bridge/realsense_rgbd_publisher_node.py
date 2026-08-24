import time

import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class RealSenseRgbdPublisherNode(Node):
    def __init__(self):
        super().__init__("realsense_rgbd_publisher_node")

        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("frame_id", "camera_color_optical_frame")

        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(Image, self.color_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "No RealSense device found. Check lsusb / USB 3.0 / udev rules."
            )

        self.device_serial = devices[0].get_info(rs.camera_info.serial_number)
        self.device_name = devices[0].get_info(rs.camera_info.name)

        self.get_logger().info(f"RealSense device: {self.device_name}")
        self.get_logger().info(f"Serial: {self.device_serial}")

        self.config.enable_device(self.device_serial)
        self.config.enable_stream(
            rs.stream.color,
            self.camera_width,
            self.camera_height,
            rs.format.bgr8,
            self.camera_fps,
        )
        self.config.enable_stream(
            rs.stream.depth,
            self.camera_width,
            self.camera_height,
            rs.format.z16,
            self.camera_fps,
        )

        self.get_logger().info("Waiting for camera stabilization...")
        time.sleep(3.0)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.get_logger().info(f"RealSense depth_scale: {self.depth_scale}")

        self.color_intr = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self.published_frame_count = 0
        self.fps_window_start = time.perf_counter()
        self.published_fps = 0.0

        self.timer = self.create_timer(1.0 / self.camera_fps, self.timer_callback)
        self.get_logger().info(f"Publishing color: {self.color_topic}")
        self.get_logger().info(f"Publishing aligned depth: {self.depth_topic}")
        self.get_logger().info(f"Publishing camera info: {self.camera_info_topic}")

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                self.get_logger().warning("Missing color/depth frame")
                return

            stamp = self.get_clock().now().to_msg()
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            color_msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            camera_info_msg = self.make_camera_info_msg()

            for msg in (color_msg, depth_msg, camera_info_msg):
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id

            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            self.camera_info_pub.publish(camera_info_msg)
            self.update_publish_fps()

        except Exception as e:
            self.get_logger().error(f"RealSense publish failed: {e}")

    def update_publish_fps(self) -> None:
        self.published_frame_count += 1
        elapsed = time.perf_counter() - self.fps_window_start
        if elapsed < 1.0:
            return
        self.published_fps = self.published_frame_count / elapsed
        self.published_frame_count = 0
        self.fps_window_start = time.perf_counter()
        self.get_logger().info(f"RGB-D publish FPS: {self.published_fps:.2f}")

    def make_camera_info_msg(self) -> CameraInfo:
        msg = CameraInfo()
        msg.width = int(self.color_intr.width)
        msg.height = int(self.color_intr.height)
        msg.distortion_model = "plumb_bob"
        msg.d = list(self.color_intr.coeffs)
        msg.k = [
            float(self.color_intr.fx),
            0.0,
            float(self.color_intr.ppx),
            0.0,
            float(self.color_intr.fy),
            float(self.color_intr.ppy),
            0.0,
            0.0,
            1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            float(self.color_intr.fx),
            0.0,
            float(self.color_intr.ppx),
            0.0,
            0.0,
            float(self.color_intr.fy),
            float(self.color_intr.ppy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return msg

    def destroy_node(self):
        self.get_logger().info("Stopping RealSense pipeline...")
        try:
            self.pipeline.stop()
            time.sleep(1.0)
        except Exception as e:
            self.get_logger().warning(f"pipeline.stop() failed: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RealSenseRgbdPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("KeyboardInterrupt. Stopped.")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
