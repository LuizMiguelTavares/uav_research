import argparse
import time

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Isaac Sim launcher
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Control the ARL Robot 1 from ROS 2 /cmd_vel."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require Isaac Sim to be running
# -----------------------------------------------------------------------------

import numpy as np
import torch
import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg
from pxr import Usd, UsdGeom
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Imu, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.sensors import CameraCfg, ImuCfg

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
from isaaclab_contrib.controllers import LeeVelControllerCfg

import omni.graph.core as og
import omni.kit.app
import omni.timeline
import isaacsim.core.experimental.utils.app as app_utils

import carb



# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

@configclass
class Ros2DroneSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    depth_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DepthTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 2.0, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(3.0, 0.0, 1.0),
        ),
    )

    robot = ARL_ROBOT_1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.01,  # 100 Hz em simulation time
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        debug_vis=False,
    )

    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_depth_camera",

        # 20 Hz em simulation time
        update_period=0.05,

        height=240,
        width=320,

        data_types=["depth"],

        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),

        offset=CameraCfg.OffsetCfg(
            # Um pouco à frente do centro do drone.
            pos=(0.15, 0.0, 0.0),

            # Com convention="world", identidade aponta para +X,
            # que estamos assumindo como a frente do ARL.
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
    )


# -----------------------------------------------------------------------------
# ROS 2 node
# -----------------------------------------------------------------------------

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

        # State exposed for a future observation term. A scan remains usable
        # after publication, while lidar_new_scan is true for one physics step.
        self.latest_valid_lidar_observation = None
        self.latest_valid_lidar_stamp = None
        self.latest_valid_lidar_age_s = float("inf")
        self.lidar_observation_valid = False
        self.lidar_new_scan = False

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
    
    def update_lidar_observation_age(self, stamp: TimeMsg) -> None:
        """Update validity state using the current simulation timestamp."""
        self.lidar_new_scan = False
        if self.latest_valid_lidar_stamp is None:
            return
        now_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        scan_ns = (
            self.latest_valid_lidar_stamp.sec * 1_000_000_000
            + self.latest_valid_lidar_stamp.nanosec
        )
        self.latest_valid_lidar_age_s = max(0.0, (now_ns - scan_ns) * 1.0e-9)

    def publish_valid_lidar(self, points_xyz: np.ndarray, stamp: TimeMsg) -> None:
        """Publish and retain one validated LiDAR revolution."""
        header = Header(stamp=stamp, frame_id="base_scan")
        msg = point_cloud2.create_cloud_xyz32(header, points_xyz)
        self.pointcloud_publisher.publish(msg)

        self.latest_valid_lidar_observation = points_xyz
        self.latest_valid_lidar_stamp = stamp
        self.latest_valid_lidar_age_s = 0.0
        self.lidar_observation_valid = True
        self.lidar_new_scan = True

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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        use_newton_actuators=False,
        render_interval=5,  # Explicit render scheduling below is authoritative.
    )

    sim = SimulationContext(sim_cfg)

    scene_cfg = Ros2DroneSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    # Start the drone above the ground.
    scene_cfg.robot.init_state.pos = (0.0, 0.0, 1.0)

    # Required by ThrusterCfg in standalone usage.
    scene_cfg.robot.actuators["thrusters"].dt = sim_cfg.dt

    scene = InteractiveScene(scene_cfg)

    from isaacsim.sensors.experimental.rtx import (
        Lidar,
        LidarSensor,
        parse_generic_model_output_data,
    )

    extension_manager = omni.kit.app.get_app().get_extension_manager()

    class LidarRevolutionAccumulator:
        """Assemble raw GMO segments between verified OS1 tick boundaries."""

        def __init__(self):
            self.last_frame_id = None
            self.synchronized = False
            self.segments = []
            self.raw_segments_received = 0
            self.raw_scans_received = 0
            self.raw_scans_rejected = 0
            self.valid_scans = 0
            self.valid_point_min = None
            self.valid_point_max = None
            self.valid_point_sum = 0

        @staticmethod
        def _xyz(gmo) -> np.ndarray:
            coordinates = np.column_stack((gmo.x, gmo.y, gmo.z)).astype(
                np.float32, copy=True
            )
            # GMO CoordsType: CARTESIAN=0, SPHERICAL=1. RTX LiDAR normally
            # supplies spherical azimuth/elevation/range coordinates.
            if int(gmo.elementsCoordsType) == 0:
                return coordinates

            azimuth = np.deg2rad(coordinates[:, 0])
            elevation = np.deg2rad(coordinates[:, 1])
            distance = coordinates[:, 2]
            horizontal_distance = distance * np.cos(elevation)
            return np.column_stack(
                (
                    horizontal_distance * np.cos(azimuth),
                    horizontal_distance * np.sin(azimuth),
                    distance * np.sin(elevation),
                )
            ).astype(np.float32, copy=False)

        def _lose_synchronization(self) -> None:
            self.synchronized = False
            self.segments.clear()

        def add(self, gmo) -> np.ndarray | None:
            """Return XYZ for one validated revolution."""
            frame_id = int(gmo.frameId)
            if frame_id == self.last_frame_id:
                return None

            frame_contiguous = (
                self.last_frame_id is None or frame_id == self.last_frame_id + 1
            )
            self.last_frame_id = frame_id
            self.raw_segments_received += 1
            if not frame_contiguous:
                self._lose_synchronization()

            points = self._xyz(gmo)
            ticks = np.asarray(gmo.tickId, dtype=np.int64).copy()
            scan_complete = bool(gmo.scanComplete)
            if len(points) != len(ticks):
                if scan_complete:
                    self.raw_scans_received += 1
                    self.raw_scans_rejected += 1
                self._lose_synchronization()
                return None

            wraps = np.flatnonzero(np.diff(ticks) < 0)
            has_verified_boundary = (
                scan_complete
                and len(wraps) == 1
                and ticks[wraps[0]] == 511
                and ticks[wraps[0] + 1] == 0
            )
            metadata_invalid = (scan_complete and not has_verified_boundary) or (
                not scan_complete and len(wraps) != 0
            )
            if metadata_invalid:
                if scan_complete:
                    self.raw_scans_received += 1
                    self.raw_scans_rejected += 1
                self._lose_synchronization()
                return None

            if not has_verified_boundary:
                if self.synchronized:
                    self.segments.append(points)
                return None

            self.raw_scans_received += 1
            split = int(wraps[0]) + 1
            completed_scan = None
            if self.synchronized:
                completed_scan = np.concatenate((*self.segments, points[:split]))
            else:
                self.raw_scans_rejected += 1

            # The remainder belongs to the next revolution. This verified wrap
            # is also the synchronization point after startup or a dropped frame.
            self.segments = [points[split:]]
            self.synchronized = True

            if completed_scan is None or len(completed_scan) == 0:
                return None

            point_count = len(completed_scan)
            self.valid_scans += 1
            self.valid_point_sum += point_count
            self.valid_point_min = (
                point_count
                if self.valid_point_min is None
                else min(self.valid_point_min, point_count)
            )
            self.valid_point_max = (
                point_count
                if self.valid_point_max is None
                else max(self.valid_point_max, point_count)
            )
            return completed_scan

    extension_manager.set_extension_enabled_immediate(
        "isaacsim.sensors.rtx.nodes",
        True,
    )

    app_utils.enable_extension("isaacsim.ros2.bridge")

    # Dá um frame para o Kit terminar de registrar os writers ROS.
    simulation_app.update()

    lidar = Lidar.create(
        path="/World/envs/env_0/Robot/base_link/os1_lidar",
        config="OS1",
        variant="OS1_REV6_32ch10hz512res",
        translations=np.array([0.0, 0.0, 0.18]),
        tick_rate=10.0,
        accumulate_outputs=False,
        aux_output_level="BASIC",
    )

    lidar_sensor = LidarSensor(
        lidar,
        annotators=["generic-model-output"],
    )

    _raw_pointcloud_writer = lidar_sensor.attach_writer(
        "RtxLidarROS2PublishPointCloud",
        topicName="point_cloud_raw",
        frameId="base_scan",
    )

    print("[INFO] Raw RTX LiDAR writer attached to /point_cloud_raw")

    # Official physics-step clock graph; shares Isaac time with the RTX ROS writer.
    clock_graph_path = "/ROS_Clock"
    keys = og.Controller.Keys
    og.Controller.edit(
        {
            "graph_path": clock_graph_path,
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPhysicsStep.outputs:step", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [("ReadSimTime.inputs:resetOnStop", True)],
        },
    )
    simulation_time = og.Controller.attribute(clock_graph_path + "/ReadSimTime.outputs:simulationTime")

    # Keep the Kit timeline from looping backwards during RTX LiDAR scan accumulation.
    omni.timeline.get_timeline_interface().set_end_time(3600.0)
    sim.reset()

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "MinimalRendering")

    robot = scene["robot"]
    imu_sensor = scene["imu"]
    depth_camera = scene["depth_camera"]
    
    # -------------------------------------------------------------------------
    # Lee velocity controller
    # -------------------------------------------------------------------------

    # Fixed gains for a deterministic A/B test with and without the camera.
    # The original gain ranges are randomized at controller.reset().
    controller_cfg = LeeVelControllerCfg(
        K_vel_range=((3.0, 3.0, 1.75), (3.0, 3.0, 1.75)),
        K_rot_range=((1.725, 1.725, 0.325), (1.725, 1.725, 0.325)),
        K_angvel_range=((0.45, 0.45, 0.0825), (0.45, 0.45, 0.0825)),
        max_inclination_angle_rad=1.0471975511965976,
        max_yaw_rate=1.0471975511965976,
    )

    controller = controller_cfg.class_type(
        cfg=controller_cfg,
        asset=robot,
        num_envs=1,
        device=args_cli.device,
    )

    controller.reset()

    # Wrench -> rotor thrust allocation.
    allocation_pinv = torch.linalg.pinv(robot.allocation_matrix)

    # -------------------------------------------------------------------------
    # ROS 2
    # -------------------------------------------------------------------------

    rclpy.init()
    ros_node = CmdVelNode()
    ros_node.publish_sensor_static_tfs(lidar.prims[0], scene_cfg)
    lidar_accumulator = LidarRevolutionAccumulator()

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    sim_dt = sim.get_physics_dt()

    # Physics and inner Lee controller both run at 100 Hz.
    # ROS /cmd_vel is only the high-level velocity setpoint; the latest setpoint
    # is held between ROS messages while Lee recomputes the wrench every physics step.
    depth_decimation = 5  # 20 Hz depth acquisition
    render_period_s = 1.0 / 40.0
    next_render_time_s = render_period_s

    step = 0

    print()
    print("==============================================")
    print(" ROS 2 ARL drone control")
    print("==============================================")
    print("Subscribed topic: /cmd_vel")
    print()
    print("linear.x  -> vx")
    print("linear.y  -> vy")
    print("linear.z  -> vz")
    print("angular.z -> yaw_rate")
    print()
    print("Command timeout: 0.5 s")
    print("==============================================")
    print()

    while simulation_app.is_running():

        # Process pending ROS callbacks without blocking simulation.
        rclpy.spin_once(
            ros_node,
            timeout_sec=0.0,
        )

        # ---------------------------------------------------------------------
        # Inner Lee controller update at every physics step (100 Hz)
        # ---------------------------------------------------------------------

        command_list = ros_node.get_command(timeout_s=0.5)

        command = torch.tensor(
            [command_list],
            dtype=torch.float32,
            device=args_cli.device,
        )

        # [vx, vy, vz, yaw_rate] -> body wrench.
        # Recompute from the CURRENT simulated state every 0.01 s.
        wrench = controller.compute(command)

        # Body wrench -> four rotor thrusts.
        thrust_target = torch.matmul(
            allocation_pinv,
            wrench.T,
        ).T

        robot.set_thrust_target(thrust_target)

        # Print once per simulated second.
        if step % 100 == 0:

            pos = robot.data.root_pos_w.torch[0]
            vel = robot.data.root_lin_vel_w.torch[0]

            print(
                f"cmd="
                f"({command_list[0]:+.2f}, "
                f"{command_list[1]:+.2f}, "
                f"{command_list[2]:+.2f}, "
                f"{command_list[3]:+.2f}) | "
                f"pos="
                f"({pos[0]:+.2f}, "
                f"{pos[1]:+.2f}, "
                f"{pos[2]:+.2f}) | "
                f"vel="
                f"({vel[0]:+.2f}, "
                f"{vel[1]:+.2f}, "
                f"{vel[2]:+.2f})"
            )

        # ---------------------------------------------------------------------
        # Physics
        # ---------------------------------------------------------------------

        scene.write_data_to_sim()

        # Physics stays at 100 Hz. Do not force RTX rendering at every physics step.
        sim.step(render=False)

        # Refresh state and publish the current-step TF before RTX writers run.
        scene.update(sim_dt)
        stamp = Time(seconds=float(simulation_time.get())).to_msg()
        ros_node.publish_robot_tf(robot, stamp)
        ros_node.update_lidar_observation_age(stamp)

        # RTX renders at 40 Hz. Depth acquisition remains independently gated at 20 Hz.
        depth_tick = ((step + 1) % depth_decimation == 0)
        current_step_time_s = (step + 1) * sim_dt
        render_tick = current_step_time_s + 1.0e-9 >= next_render_time_s
        if render_tick:
            next_render_time_s += render_period_s
            if sim.is_rendering:
                sim.render()
                data, _ = lidar_sensor.get_data("generic-model-output")
                if data is not None:
                    gmo = parse_generic_model_output_data(data)
                    rejected_before = lidar_accumulator.raw_scans_rejected
                    validated = lidar_accumulator.add(gmo)
                    if validated is not None:
                        points_xyz = validated
                        # Match the official RTX ROS writer: stamp a newly completed
                        # scan with Isaac simulation time from this render.
                        ros_node.publish_valid_lidar(points_xyz, stamp)
                        print(
                            f"LIDAR_VALID | points={len(points_xyz)} | "
                            f"frame={gmo.frameId}",
                            flush=True,
                        )
                    elif lidar_accumulator.raw_scans_rejected > rejected_before:
                        print(
                            f"LIDAR_REJECTED | frame={gmo.frameId} | "
                            f"scanComplete={int(gmo.scanComplete)}",
                            flush=True,
                        )

        if step % 2 == 0:
            ros_node.publish_odom(robot, stamp)

        ros_node.publish_imu(imu_sensor, stamp)

        # Request the depth image only when a new 20 Hz frame is due.
        if depth_tick:
            depth = depth_camera.data.output["depth"].torch

            depth_img = depth[0, ..., 0]
            intrinsic_matrix = depth_camera.data.intrinsic_matrices.torch[0]

            ros_node.publish_depth(depth_img, intrinsic_matrix, stamp)

            if (step + 1) % 100 == 0:
                finite = depth_img[torch.isfinite(depth_img)]

                if finite.numel() > 0:
                    print(
                        f"DEPTH shape={tuple(depth_img.shape)} | "
                        f"min={finite.min().item():.2f} m | "
                        f"max={finite.max().item():.2f} m | "
                        f"finite={100.0 * finite.numel() / depth_img.numel():.1f}%"
                    )

        # ---------------------------------------------------------------------
        # Follow camera
        # ---------------------------------------------------------------------

        # if step % 5 == 0:

        #     pos = robot.data.root_pos_w.torch[0]

        #     sim.set_camera_view(
        #         eye=[
        #             float(pos[0] + 4.0),
        #             float(pos[1] + 4.0),
        #             float(pos[2] + 2.5),
        #         ],
        #         target=[
        #             float(pos[0]),
        #             float(pos[1]),
        #             float(pos[2]),
        #         ],
        #     )

        step += 1

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    valid_mean = (
        lidar_accumulator.valid_point_sum / lidar_accumulator.valid_scans
        if lidar_accumulator.valid_scans
        else 0.0
    )
    print(
        "LIDAR_SUMMARY | "
        f"raw_segments={lidar_accumulator.raw_segments_received} | "
        f"raw_scans={lidar_accumulator.raw_scans_received} | "
        f"rejected={lidar_accumulator.raw_scans_rejected} | "
        f"valid={lidar_accumulator.valid_scans} | "
        f"points_min={lidar_accumulator.valid_point_min or 0} | "
        f"points_mean={valid_mean:.2f} | "
        f"points_max={lidar_accumulator.valid_point_max or 0}",
        flush=True,
    )

    ros_node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()