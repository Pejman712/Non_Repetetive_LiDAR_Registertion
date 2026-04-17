#!/usr/bin/env python3
"""
ROS2 Odometry -> TUM trajectory writer (compatible with your "my_node" YAML namespace)

This node subscribes to nav_msgs/Odometry and writes a .tum file:
timestamp tx ty tz qx qy qz qw

Key change: parameters can live under your existing YAML node key:
  my_node:
    ros__parameters:
      odom_to_tum:
        enabled: true
        odom_topic: "/lidar/odom"
        output_path: "/tmp/lidar_odom.tum"
        flush_every_n: 10
        use_msg_time: true
        append: true

Run:
  ros2 run <your_pkg> odom_to_tum.py --ros-args --params-file config/my_config.yaml -r __node:=my_node

Notes:
- The node name must match the YAML key ("my_node") to automatically load those params.
- If odom_to_tum.enabled is false, the node will not subscribe / write.
"""

import os
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def stamp_to_float_seconds(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class OdomToTUM(Node):
    def __init__(self):
        super().__init__("my_node")  # IMPORTANT: match your YAML key

        # ---- Declare root params so ROS2 loads your file without warnings
        # (These can exist in your big config; we don't use them here.)
        self.declare_parameter("lidar_topic", "/points")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("step_decimation", 1)
        self.declare_parameter("max_scans", -1)
        self.declare_parameter("accumulate_between_decimation", False)
        self.declare_parameter("accumulate_voxel", 0.1)
        self.declare_parameter("accumulate_max_points", 1_500_000)
        self.declare_parameter("force_z_zero", False)
        self.declare_parameter("z_redistribution_method", "prediction")
        self.declare_parameter("fixed_weights", False)
        self.declare_parameter("feature_weight", 0.3)
        self.declare_parameter("geometric_weight", 0.4)
        self.declare_parameter("temporal_weight", 0.3)
        self.declare_parameter("freeze_adaptation", False)
        self.declare_parameter("visualize", False)
        self.declare_parameter("map_voxel", 0.15)
        self.declare_parameter("gicp.max_corr_distance", 2.0)
        self.declare_parameter("gicp.voxel_size", 0.2)
        self.declare_parameter("gicp.max_iterations", 50)

        # ---- Our nested config under your "my_node"
        def gp(name: str, default):
            # nested group: odom_to_tum.<name>
            full = f"odom_to_tum.{name}"
            self.declare_parameter(full, default)
            return self.get_parameter(full).value

        self.enabled = bool(gp("enabled", True))
        self.odom_topic = str(gp("odom_topic", "/lidar/odom"))
        self.output_path = str(gp("output_path", "/tmp/lidar_odom.tum"))
        self.flush_every_n = max(1, int(gp("flush_every_n", 10)))
        self.use_msg_time = bool(gp("use_msg_time", True))
        self.append = bool(gp("append", True))

        self._fh = None
        self._count = 0
        self.sub = None

        if not self.enabled:
            self.get_logger().info("odom_to_tum.enabled=false -> not subscribing / not writing.")
            return

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        mode = "a" if self.append else "w"
        self._fh = open(self.output_path, mode, buffering=1)  # line-buffered

        self.sub = self.create_subscription(Odometry, self.odom_topic, self.cb, 50)

        self.get_logger().info(f"[odom_to_tum] Writing TUM to: {self.output_path} (mode={mode})")
        self.get_logger().info(f"[odom_to_tum] Subscribing to: {self.odom_topic}")
        self.get_logger().info(f"[odom_to_tum] use_msg_time={self.use_msg_time} flush_every_n={self.flush_every_n}")

    def cb(self, msg: Odometry):
        if self._fh is None:
            return

        if self.use_msg_time:
            t = stamp_to_float_seconds(msg.header.stamp)
        else:
            t = self.get_clock().now().nanoseconds * 1e-9

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        line = (
            f"{t:.18e} "
            f"{float(p.x):.18e} {float(p.y):.18e} {float(p.z):.18e} "
            f"{float(q.x):.18e} {float(q.y):.18e} {float(q.z):.18e} {float(q.w):.18e}\n"
        )

        try:
            self._fh.write(line)
            self._count += 1
            if (self._count % self.flush_every_n) == 0:
                self._fh.flush()
                os.fsync(self._fh.fileno())
        except Exception as e:
            self.get_logger().error(f"[odom_to_tum] Failed writing to {self.output_path}: {e}")

    def destroy_node(self):
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node: Optional[OdomToTUM] = None
    try:
        node = OdomToTUM()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()