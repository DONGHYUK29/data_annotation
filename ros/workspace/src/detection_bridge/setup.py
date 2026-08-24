import glob
import os

from setuptools import find_packages, setup

package_name = 'detection_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='prml513',
    maintainer_email='sangwoo4876@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'yolo_realsense_node = detection_bridge.yolo_realsense_node:main',
        'yolo_realsense_depth_refine_node = detection_bridge.yolo_realsense_depth_refine_node:main',
        'realsense_rgbd_publisher_node = detection_bridge.realsense_rgbd_publisher_node:main',
        'annotated_image_viewer_node = detection_bridge.annotated_image_viewer_node:main',
    ],
},
)
