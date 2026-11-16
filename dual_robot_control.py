#!/usr/bin/env python3
"""
Task 2: Control both robots (ego + opponent) from one Python ROS 2 node.

Topics used (f1tenth_gym_ros defaults, overridable as ROS parameters):
    Robot 1 (ego)  command : /drive       (ackermann_msgs/AckermannDriveStamped)
    Robot 1 (ego)  odometry: /ego_racecar/odom (nav_msgs/Odometry)
    Robot 2 (opp)  command : /opp_drive   (ackermann_msgs/AckermannDriveStamped)
    Robot 2 (opp)  odometry: /opp_racecar/odom (nav_msgs/Odometry)

Requires num_agent: 2 in sim.yaml.

Run:
    python3 dual_robot_control.py
    python3 dual_robot_control.py --ros-args -p ego_odom_topic:=/ego_racecar/odom
"""

import math
import time

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


# Each segment: (name, duration in seconds, speed in m/s, steering angle in rad)
# The two robots deliberately follow DIFFERENT sequences to show independence.
EGO_SEQUENCE = [
    ("forward", 3.0, 1.0, 0.0),
    ("turn left", 3.0, 1.0, 0.3),
    ("forward", 2.0, 1.0, 0.0),
    ("stop", 1.0, 0.0, 0.0),
]

OPP_SEQUENCE = [
    ("forward", 2.0, 1.5, 0.0),
    ("turn right", 3.0, 1.0, -0.3),
    ("forward", 3.0, 1.5, 0.0),
    ("stop", 1.0, 0.0, 0.0),
]


def yaw_from_quaternion(q):
    """Extract yaw (rad) from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RobotChannel:
    """Everything belonging to ONE robot: its publisher, odometry, and sequence."""

    def __init__(self, node, label, cmd_topic, odom_topic, sequence):
        self.label = label
        self.cmd_topic = cmd_topic
        self.odom_topic = odom_topic
        self.sequence = sequence
        self.total_time = sum(seg[1] for seg in sequence)

        self.pub = node.create_publisher(AckermannDriveStamped, cmd_topic, 10)
        self.sub = node.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )

        self.start_pose = None   # (x, y, yaw) at first odometry message
        self.pose = None         # latest (x, y, yaw)
        self.speed = 0.0         # latest linear speed from odometry
        self.distance = 0.0      # accumulated path length
        self._last_xy = None
        self.current_segment = None

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (p.x, p.y, yaw)
        self.speed = msg.twist.twist.linear.x
        if self.start_pose is None:
            self.start_pose = self.pose
        if self._last_xy is not None:
            self.distance += math.hypot(p.x - self._last_xy[0],
                                        p.y - self._last_xy[1])
        self._last_xy = (p.x, p.y)

    def segment_at(self, t):
        """Return the (name, speed, steering) active at elapsed time t, or None."""
        elapsed = 0.0
        for name, duration, speed, steer in self.sequence:
            if t < elapsed + duration:
                return name, speed, steer
            elapsed += duration
        return None

    def send(self, node, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.pub.publish(msg)

    def stop(self, node):
        self.send(node, 0.0, 0.0)

    def summary(self):
        if self.start_pose is None or self.pose is None:
            return f"{self.label}: NO ODOMETRY RECEIVED on {self.odom_topic}"
        dx = self.pose[0] - self.start_pose[0]
        dy = self.pose[1] - self.start_pose[1]
        dyaw = math.degrees(self.pose[2] - self.start_pose[2])
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        return (f"{self.label}: displacement={math.hypot(dx, dy):.2f} m, "
                f"path length={self.distance:.2f} m, "
                f"heading change={dyaw:.1f} deg, "
                f"final speed={self.speed:.2f} m/s")


class DualRobotController(Node):
    def __init__(self):
        super().__init__("dual_robot_controller")

        # Topics are parameters so they can be changed without editing code.
        self.declare_parameter("ego_cmd_topic", "/drive")
        self.declare_parameter("ego_odom_topic", "/ego_racecar/odom")
        self.declare_parameter("opp_cmd_topic", "/opp_drive")
        self.declare_parameter("opp_odom_topic", "/opp_racecar/odom")
        self.declare_parameter("rate_hz", 20.0)

        p = self.get_parameter
        self.ego = RobotChannel(self, "EGO", p("ego_cmd_topic").value,
                                p("ego_odom_topic").value, EGO_SEQUENCE)
        self.opp = RobotChannel(self, "OPP", p("opp_cmd_topic").value,
                                p("opp_odom_topic").value, OPP_SEQUENCE)
        self.robots = [self.ego, self.opp]

        self.get_logger().info("Topic configuration:")
        for r in self.robots:
            self.get_logger().info(
                f"  {r.label}: publishes {r.cmd_topic}  |  monitors {r.odom_topic}")

        self.t0 = None
        self.finished = False
        self._last_log = 0.0
        rate = p("rate_hz").value
        self.timer = self.create_timer(1.0 / rate, self.tick)

    def tick(self):
        if self.finished:
            return

        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
            self.get_logger().info("Starting movement sequences")
        t = now - self.t0

        # Each robot is commanded independently from its own sequence.
        for r in self.robots:
            seg = r.segment_at(t)
            if seg is None:
                r.stop(self)
                continue
            name, speed, steer = seg
            if name != r.current_segment:
                r.current_segment = name
                self.get_logger().info(
                    f"[{r.label}] t={t:.1f}s -> {name} "
                    f"(speed={speed} m/s, steering={steer} rad)")
            r.send(self, speed, steer)

        # Odometry monitoring once per second.
        if t - self._last_log >= 1.0:
            self._last_log = t
            for r in self.robots:
                if r.pose is None:
                    self.get_logger().warn(
                        f"[{r.label}] no odometry yet on {r.odom_topic}")
                else:
                    self.get_logger().info(
                        f"[{r.label}] odom x={r.pose[0]:.2f} y={r.pose[1]:.2f} "
                        f"yaw={math.degrees(r.pose[2]):.1f} deg "
                        f"v={r.speed:.2f} m/s")

        # Both sequences done (plus 1 s of holding zero velocity).
        if t > max(r.total_time for r in self.robots) + 1.0:
            self.finished = True

    def safe_stop(self, repeats=10):
        """Send zero velocity to BOTH robots several times so it is not missed."""
        self.get_logger().info("Sending zero velocity to both robots")
        for _ in range(repeats):
            for r in self.robots:
                r.stop(self)
            time.sleep(0.05)

    def report(self):
        self.get_logger().info("=== Odometry verification ===")
        for r in self.robots:
            self.get_logger().info(r.summary())


def main():
    rclpy.init()
    node = DualRobotController()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        # Runs on normal finish, Ctrl+C, or any exception.
        node.safe_stop()
        # Spin briefly so the last odometry messages arrive before reporting.
        end = time.monotonic() + 0.5
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
