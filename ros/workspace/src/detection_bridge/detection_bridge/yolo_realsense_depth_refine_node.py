import rclpy

from detection_bridge.ros_yolo_base_node import RosYoloBaseNode


class YoloRealSenseDepthRefineNode(RosYoloBaseNode):
    def __init__(self):
        super().__init__(
            node_name="yolo_realsense_depth_refine_node",
            default_use_depth_refine=True,
            window_name="RealSense YOLO Depth Refine ROS",
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = YoloRealSenseDepthRefineNode()
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
