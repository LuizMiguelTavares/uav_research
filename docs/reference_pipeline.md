# Validated reference pipeline

This pipeline is a system-validation fixture. ARL Robot 1 and the Ouster OS1 profile are not project-wide platform commitments.

## Validated toolchain

- Ubuntu 24.04
- Python 3.12
- Isaac Sim 6.1 GA
- Isaac Lab 3.0.0-EA source commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`
- ROS 2 Jazzy
- PhysX
- NVIDIA RTX 4070 Laptop GPU

The migration source was `IsaacLab/scripts/research_spike/validate_ros2_cmd_vel.py`, SHA-256 `462f7d397a3882c0c75d97eddfcd8ad114318465bdd1fb01fde5d97562c8319e`. The upstream checkout remains an external, read-only dependency at `../IsaacLab`.

## Timing and frames

| Signal | Simulation cadence |
|---|---:|
| Physics | 100 Hz |
| Lee velocity controller feedback | 100 Hz |
| IMU | 100 Hz |
| Odometry | 50 Hz |
| RTX rendering | 40 Hz |
| Depth acquisition | 20 Hz |
| Validated OS1 revolution | approximately 10 Hz |

The loop writes actuator targets, steps physics without implicit rendering, refreshes the scene state, publishes dynamic TF, and then performs explicitly scheduled RTX rendering. Depth acquisition is decimated independently from rendering.

Isaac simulation time is the single timestamp source. The official ROS clock graph publishes `/clock`. Dynamic TF is published before RTX writers run. The frame tree is:

```text
world -> base_link -> base_scan
                   -> imu_link
                   -> front_depth_camera
```

The camera frame uses ROS optical-axis convention. Depth images and `CameraInfo` share the same timestamp and `front_depth_camera` frame.

## Run

The first invocation from a newly created environment may ask the user to accept NVIDIA's Isaac Sim EULA.

```bash
cd ~/research/uav_research
uv sync --extra isaacsim
uv run --extra isaacsim python scripts/standalone/ros2_system_validation.py --visualizer kit
```

For a bounded integration run:

```bash
uv run --extra isaacsim python scripts/standalone/ros2_system_validation.py \
  --visualizer kit --max_steps 600
```

The default has no step limit and preserves the interactive spike behavior.

## Depth point cloud

In another ROS-sourced terminal:

```bash
ros2 run depth_image_proc point_cloud_xyz_node --ros-args \
  -p use_sim_time:=true \
  -r image_rect:=/camera/depth/image_raw \
  -r camera_info:=/camera/depth/camera_info \
  -r points:=/camera/depth/points
```

RViz settings:

- `use_sim_time=true`
- Fixed Frame: `world`
- RTX LiDAR PointCloud2: `/point_cloud`
- Depth PointCloud2: `/camera/depth/points`

## Validation commands

```bash
ros2 topic hz /point_cloud
ros2 topic hz /camera/depth/image_raw
ros2 topic hz /camera/depth/camera_info
ros2 topic hz /camera/depth/points
ros2 topic hz /imu
ros2 topic hz /odom
ros2 topic echo /point_cloud --once --field header
ros2 topic echo /camera/depth/image_raw --once --field header
ros2 topic echo /camera/depth/camera_info --once --field header
ros2 run tf2_ros tf2_echo world base_scan
ros2 run tf2_ros tf2_echo world front_depth_camera
```

`ros2 topic hz` measures wall-clock arrival rate. It does not directly prove simulation-time cadence when the simulator runs slower or faster than real time. During migration, `/point_cloud` remained approximately 10 Hz wall time. The full visualization stack produced lower, jittery depth-derived point-cloud wall rates, consistent with the known pre-migration performance observation.

## Current limitations

- The zero-velocity reference settles close to ground contact in the current scene; a positive vertical `/cmd_vel` command was verified to produce upward motion. Controller behavior was preserved from the spike.
- The MotionBVH warning remains. It concerns motion effects, not the previously diagnosed scan truncation.
- The renderer warns about Fabric transforms with geometry streaming. The validated pipeline retains the working configuration.
- The Kit timeline end time remains fixed at 3600 seconds to prevent timeline wrap from stopping RTX LiDAR output.
