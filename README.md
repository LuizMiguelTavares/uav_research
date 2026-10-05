# UAV Research

Research software for reproducible UAV simulation, reinforcement learning, model predictive control (MPC), motion planning, perception, and eventual real-hardware validation. The repository is an external Isaac Lab project and is intentionally not tied to a final vehicle, sensor suite, learning method, or planning method.

The initial reference pipeline preserves a validated Isaac Sim 6.1 / Isaac Lab 3.0.0-EA configuration using ARL Robot 1, ROS 2 Jazzy, depth sensing, and RTX LiDAR. These are validation fixtures rather than project-wide platform choices.

## Setup

The project currently expects the Isaac Lab checkout at `../IsaacLab` and records its Python dependencies in `uv.lock`.

```bash
cd ~/research/uav_research
source /opt/ros/jazzy/setup.bash
uv sync --extra isaacsim
```

Source the ROS 2 Jazzy setup in each fresh shell before running the ROS reference pipeline. The first Isaac Sim invocation from a new environment may prompt you to accept NVIDIA's EULA.

## Reference simulation

```bash
uv run --extra isaacsim python scripts/standalone/ros2_system_validation.py --visualizer kit
```

See [the reference pipeline](docs/reference_pipeline.md) for ROS topics, `depth_image_proc`, RViz, timing, frames, and validation commands. See [the RTX LiDAR compatibility note](docs/rtx_lidar_compatibility.md) before changing LiDAR accumulation.

## Development

```bash
uv run pytest tests/unit
uv run ruff check .
```

Future Isaac Lab tasks belong under `src/uav_research/tasks` using the installed manager-based task conventions. No scientific task is registered yet because the first RL/MPC research formulation remains open.

This private repository intentionally has no project license yet.
