"""ROS 2 message and TF bridge for the validated reference pipeline."""

import time

import numpy as np
import torch
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from pxr import Usd, UsdGeom
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .scene_cfg import Ros2DroneSceneCfg


class CmdVelNode(Node):

    def __init__(self):
        super().__init__(
            "isaac_drone_cmd_vel",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )

        self.command = [0.0, 0.0, 0.0, 0.0]
        self.last_message_time = None

        self.subscription = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        self.odom_publisher = self.create_publisher(
            Odometry,
            "/odom",
            10,
        )

        self.imu_publisher = self.create_publisher(
            Imu,
            "/imu",
            10,
        )

        self.depth_publisher = self.create_publisher(
            Image,
            "/camera/depth/image_raw",
            10,
        )

        self.depth_camera_info_publisher = self.create_publisher(
            CameraInfo,
            "/camera/depth/camera_info",
            10,
        )

        self.depth_debug_publisher = self.create_publisher(
            Image,
            "/camera/depth/debug",
            10,
        )

        self.pointcloud_publisher = self.create_publisher(
            PointCloud2,
            "/point_cloud",
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.get_logger().info("Listening to /cmd_vel")

    def cmd_vel_callback(self, msg: Twist):

        self.command = [
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.z,
        ]

        self.last_message_time = time.monotonic()

    def get_command(self, timeout_s=0.5):

        # No command received yet.
        if self.last_message_time is None:
            return [0.0, 0.0, 0.0, 0.0]

        # Deadman timeout.
        if time.monotonic() - self.last_message_time > timeout_s:
            return [0.0, 0.0, 0.0, 0.0]

        return self.command
    
    def publish_odom(self, robot, stamp: TimeMsg):

        pos = robot.data.root_pos_w.torch[0]
        quat = robot.data.root_quat_w.torch[0]

        lin_vel = robot.data.root_lin_vel_b.torch[0]
        ang_vel = robot.data.root_ang_vel_b.torch[0]

        msg = Odometry()

        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.child_frame_id = "base_link"

        # Position
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])

        # Isaac Lab 3 and ROS both use [x, y, z, w].
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])

        # Twist is expressed in child_frame_id = base_link
        msg.twist.twist.linear.x = float(lin_vel[0])
        msg.twist.twist.linear.y = float(lin_vel[1])
        msg.twist.twist.linear.z = float(lin_vel[2])

        msg.twist.twist.angular.x = float(ang_vel[0])
        msg.twist.twist.angular.y = float(ang_vel[1])
        msg.twist.twist.angular.z = float(ang_vel[2])

        self.odom_publisher.publish(msg)
    
    def publish_imu(self, imu_sensor, stamp: TimeMsg):
        data = imu_sensor.data

        ang_vel = data.ang_vel_b.torch[0]
        lin_acc = data.lin_acc_b.torch[0]

        msg = Imu()

        msg.header.stamp = stamp
        msg.header.frame_id = "imu_link"

        # Esta IMU não fornece orientação.
        msg.orientation_covariance[0] = -1.0

        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])

        msg.linear_acceleration.x = float(lin_acc[0])
        msg.linear_acceleration.y = float(lin_acc[1])
        msg.linear_acceleration.z = float(lin_acc[2])

        self.imu_publisher.publish(msg)
    
    def publish_depth(self, depth_img, intrinsic_matrix, stamp: TimeMsg):

        # ---------------------------------------------------------
        # Raw metric depth: float32, metres
        # ---------------------------------------------------------

        depth_cpu = (
            depth_img
            .contiguous()
            .to("cpu")
            .numpy()
            .astype("float32", copy=False)
        )

        raw_msg = Image()

        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = "front_depth_camera"

        raw_msg.height = depth_cpu.shape[0]
        raw_msg.width = depth_cpu.shape[1]

        raw_msg.encoding = "32FC1"
        raw_msg.is_bigendian = 0
        raw_msg.step = depth_cpu.shape[1] * 4
        raw_msg.data = depth_cpu.tobytes()

        self.depth_publisher.publish(raw_msg)

        intrinsic_cpu = intrinsic_matrix.to("cpu").numpy()
        fx = float(intrinsic_cpu[0, 0])
        fy = float(intrinsic_cpu[1, 1])
        cx = float(intrinsic_cpu[0, 2])
        cy = float(intrinsic_cpu[1, 2])

        camera_info_msg = CameraInfo()
        camera_info_msg.header = raw_msg.header
        camera_info_msg.width = raw_msg.width
        camera_info_msg.height = raw_msg.height
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.d = [0.0] * 5
        camera_info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.depth_camera_info_publisher.publish(camera_info_msg)

        # ---------------------------------------------------------
        # Debug visualization
        #
        # 0.1 m -> white
        # 5.0 m -> black
        # inf   -> black
        # ---------------------------------------------------------

        depth_vis = torch.nan_to_num(
            depth_img,
            nan=5.0,
            posinf=5.0,
            neginf=0.1,
        )

        depth_vis = torch.clamp(
            depth_vis,
            0.1,
            5.0,
        )

        depth_vis = (
            (5.0 - depth_vis)
            / (5.0 - 0.1)
            * 255.0
        ).to(torch.uint8)

        depth_vis = (
            depth_vis
            .contiguous()
            .to("cpu")
            .numpy()
        )

        debug_msg = Image()

        debug_msg.header = raw_msg.header

        debug_msg.height = depth_vis.shape[0]
        debug_msg.width = depth_vis.shape[1]

        debug_msg.encoding = "mono8"
        debug_msg.is_bigendian = 0
        debug_msg.step = depth_vis.shape[1]
        debug_msg.data = depth_vis.tobytes()

        self.depth_debug_publisher.publish(debug_msg)
    
    def publish_valid_lidar(self, points_xyz: np.ndarray, stamp: TimeMsg) -> None:
        """Publish and retain one validated LiDAR revolution."""
        header = Header(stamp=stamp, frame_id="base_scan")
        msg = point_cloud2.create_cloud_xyz32(header, points_xyz)
        self.pointcloud_publisher.publish(msg)

    def publish_sensor_static_tfs(self, lidar_prim: Usd.Prim, scene_cfg: Ros2DroneSceneCfg) -> None:
        """Publish fixed mounts, with the image frame using ROS optical axes."""
        stage = lidar_prim.GetStage()
        base_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/base_link")
        cache = UsdGeom.XformCache()
        # The referenced Ouster asset can add a transform below its mounting Xform.
        lidar_mount, _ = cache.ComputeRelativeTransform(lidar_prim, base_prim)
        lidar_quat = lidar_mount.ExtractRotationQuat()
        lidar_xyzw = (*lidar_quat.GetImaginary(), lidar_quat.GetReal())
        camera_prim = stage.GetPrimAtPath(str(base_prim.GetPath()) + "/front_depth_camera")
        camera_mount, _ = cache.ComputeRelativeTransform(camera_prim, base_prim)
        camera_quat = camera_mount.ExtractRotationQuat()
        camera_xyzw = torch.tensor([(*camera_quat.GetImaginary(), camera_quat.GetReal())])
        camera_ros = convert_camera_frame_orientation_convention(camera_xyzw, origin="opengl", target="ros")[0]
        mounts = [
            ("base_scan", lidar_mount.ExtractTranslation(), lidar_xyzw),
            ("imu_link", scene_cfg.imu.offset.pos, scene_cfg.imu.offset.rot),
            ("front_depth_camera", camera_mount.ExtractTranslation(), camera_ros),
        ]
        transforms = []
        for frame, translation, rotation in mounts:
            tf = TransformStamped()
            # Static transforms are timeless; zero is also valid before the first /clock.
            tf.header.frame_id = "base_link"
            tf.child_frame_id = frame
            tf.transform.translation.x = float(translation[0])
            tf.transform.translation.y = float(translation[1])
            tf.transform.translation.z = float(translation[2])
            tf.transform.rotation.x = float(rotation[0])
            tf.transform.rotation.y = float(rotation[1])
            tf.transform.rotation.z = float(rotation[2])
            tf.transform.rotation.w = float(rotation[3])
            transforms.append(tf)
        self.static_tf_broadcaster.sendTransform(transforms)

    def publish_robot_tf(self, robot, stamp: TimeMsg):
        """world -> base_link using the simulated drone pose."""

        pos = robot.data.root_pos_w.torch[0]
        quat = robot.data.root_quat_w.torch[0]

        tf = TransformStamped()

        tf.header.stamp = stamp

        tf.header.frame_id = "world"
        tf.child_frame_id = "base_link"

        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])

        # Isaac Lab 3 and ROS both use [x, y, z, w].
        tf.transform.rotation.x = float(quat[0])
        tf.transform.rotation.y = float(quat[1])
        tf.transform.rotation.z = float(quat[2])
        tf.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(tf)
