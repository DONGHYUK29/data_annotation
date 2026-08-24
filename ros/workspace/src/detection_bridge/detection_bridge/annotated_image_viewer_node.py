import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class AnnotatedImageViewerNode(Node):
    def __init__(self):
        super().__init__("annotated_image_viewer_node")

        self.declare_parameter("annotated_image_topic", "/annotated_image")
        self.declare_parameter("window_name", "RealSense YOLO ROS")

        self.annotated_image_topic = str(
            self.get_parameter("annotated_image_topic").value
        )
        self.window_name = str(self.get_parameter("window_name").value)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.last_displayed_seq = 0
        self.last_display_time = time.perf_counter()
        self.received_frame_count = 0
        self.displayed_new_frame_count = 0

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub = self.create_subscription(
            Image,
            self.annotated_image_topic,
            self.on_image,
            image_qos,
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 960)
        self.get_logger().info(
            f"Showing annotated image topic {self.annotated_image_topic}"
        )

    def on_image(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_frame_seq += 1
            self.received_frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to convert annotated image: {e}")

    def spin_viewer(self) -> None:
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.001)
                if self.latest_frame is not None:
                    cv2.imshow(self.window_name, self.latest_frame)
                    if self.latest_frame_seq != self.last_displayed_seq:
                        self.displayed_new_frame_count += 1
                        self.last_displayed_seq = self.latest_frame_seq

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.get_logger().info("ESC pressed. Closing viewer...")
                    break

                now = time.perf_counter()
                elapsed = now - self.last_display_time
                if elapsed >= 1.0:
                    received_fps = self.received_frame_count / elapsed
                    display_fps = self.displayed_new_frame_count / elapsed
                    self.get_logger().info(
                        f"Annotated image FPS: received={received_fps:.2f} displayed={display_fps:.2f}"
                    )
                    self.received_frame_count = 0
                    self.displayed_new_frame_count = 0
                    self.last_display_time = now
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = AnnotatedImageViewerNode()
        node.spin_viewer()
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
