from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("regnonrep")
    params = os.path.join(pkg_share, "config", "non_rep_lidar.yaml")

    return LaunchDescription([
        Node(
            package="regnonrep",
            executable="ros_non_rep.py",
            name="my_node",
            parameters=[params],
            output="screen",
        ),
        Node(
            package="regnonrep",
            executable="odom_to_tum.py",
            name="my_node",          # must match YAML key
            parameters=[params],
            output="screen",
        ),
    ])