import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Isaac Sim launcher
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Control the ARL Robot 1 from ROS 2 /cmd_vel."
)

parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many physics steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require Isaac Sim to be running
# -----------------------------------------------------------------------------

import carb
import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import rclpy
import torch
from isaaclab_contrib.controllers import LeeVelControllerCfg
from rclpy.time import Time

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from uav_research.reference.ros2_bridge import CmdVelNode
from uav_research.reference.scene_cfg import Ros2DroneSceneCfg
from uav_research.sim.rtx_lidar_accumulator import RtxLidarRevolutionAccumulator

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
    lidar_accumulator = RtxLidarRevolutionAccumulator()

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

    while simulation_app.is_running() and (args_cli.max_steps is None or step < args_cli.max_steps):

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
        simulation_time_s = float(simulation_time.get())
        stamp = Time(seconds=simulation_time_s).to_msg()
        ros_node.publish_robot_tf(robot, stamp)
        lidar_accumulator.update_age(simulation_time_s)

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
                    validated = lidar_accumulator.add(gmo, timestamp_s=simulation_time_s)
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