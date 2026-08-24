import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("detection_bridge")
    default_params_file = os.path.join(package_share, "config", "yolo_realsense.yaml")

    params_file = LaunchConfiguration("params_file")
    model_path = LaunchConfiguration("model_path")
    inference_device = LaunchConfiguration("inference_device")
    use_depth_refine = LaunchConfiguration("use_depth_refine")
    show_window = LaunchConfiguration("show_window")
    sync_slop_sec = LaunchConfiguration("sync_slop_sec")
    image_qos_reliability = LaunchConfiguration("image_qos_reliability")
    debug_stamp_delta = LaunchConfiguration("debug_stamp_delta")
    publish_annotated_image = LaunchConfiguration("publish_annotated_image")
    publish_instance_mask = LaunchConfiguration("publish_instance_mask")
    publish_json_debug = LaunchConfiguration("publish_json_debug")

    common_parameters = [
        params_file,
        {
            "model_path": model_path,
            "inference_device": ParameterValue(inference_device, value_type=str),
            "show_window": ParameterValue(show_window, value_type=bool),
            "sync_slop_sec": ParameterValue(sync_slop_sec, value_type=float),
            "image_qos_reliability": ParameterValue(
                image_qos_reliability, value_type=str
            ),
            "debug_stamp_delta": ParameterValue(debug_stamp_delta, value_type=bool),
            "publish_annotated_image": ParameterValue(
                publish_annotated_image, value_type=bool
            ),
            "publish_instance_mask": ParameterValue(
                publish_instance_mask, value_type=bool
            ),
            "publish_json_debug": ParameterValue(publish_json_debug, value_type=bool),
        },
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="YAML file containing the YOLO node parameters.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Absolute path to the YOLO segmentation .pt weights.",
            ),
            DeclareLaunchArgument(
                "inference_device",
                default_value="",
                description="Ultralytics device. Empty selects CUDA device 0 when available, otherwise CPU.",
            ),
            DeclareLaunchArgument(
                "use_depth_refine",
                default_value="false",
                description="Select the depth-refine YOLO wrapper.",
            ),
            DeclareLaunchArgument(
                "show_window",
                default_value="true",
                description="Open the OpenCV preview window.",
            ),
            DeclareLaunchArgument(
                "sync_slop_sec",
                default_value="0.08",
                description="Approximate RGB/Depth sync slop in seconds. Try 0.2 or 0.5 when stamps are loose.",
            ),
            DeclareLaunchArgument(
                "image_qos_reliability",
                default_value="reliable",
                description="Image subscription QoS: reliable or best_effort.",
            ),
            DeclareLaunchArgument(
                "debug_stamp_delta",
                default_value="false",
                description="Print RGB/Depth header stamp delta at debug log level.",
            ),
            DeclareLaunchArgument(
                "publish_annotated_image",
                default_value="false",
                description="Publish annotated preview images.",
            ),
            DeclareLaunchArgument(
                "publish_instance_mask",
                default_value="false",
                description="Publish 16-bit instance mask images.",
            ),
            DeclareLaunchArgument(
                "publish_json_debug",
                default_value="false",
                description="Publish JSON debug detections.",
            ),
            Node(
                package="detection_bridge",
                executable="yolo_realsense_node",
                name="yolo_realsense_node",
                output="screen",
                parameters=common_parameters,
                condition=UnlessCondition(use_depth_refine),
            ),
            Node(
                package="detection_bridge",
                executable="yolo_realsense_depth_refine_node",
                name="yolo_realsense_depth_refine_node",
                output="screen",
                parameters=common_parameters,
                condition=IfCondition(use_depth_refine),
            ),
        ]
    )
