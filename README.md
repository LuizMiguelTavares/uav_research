# UAV Research

Research software for reproducible UAV simulation, reinforcement learning, motion planning, perception, and eventual real-hardware validation. The repository is an external Isaac Lab project and is intentionally not tied to a final vehicle, sensor suite, learning method, or planning method.

The initial reference pipeline preserves a validated Isaac Sim 6.1 / Isaac Lab 3.0.0-EA configuration using ARL Robot 1, ROS 2 Jazzy, depth sensing, and RTX LiDAR. Those components are validation fixtures rather than project-wide platform choices.

## Local toolchain

- Ubuntu 24.04
- Python 3.12
- Isaac Sim 6.1 GA
- Isaac Lab 3.0.0-EA at commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`
- ROS 2 Jazzy
- PhysX

The generated `pyproject.toml` expects the Isaac Lab checkout at `../IsaacLab`.

## Setup

```bash
cd ~/research/uav_research
uv sync --extra isaacsim
```

## Commands

```bash
# List registered environments once project tasks exist.
uv run python scripts/list_envs.py --show_presets

# Run lightweight tests.
uv run pytest -m 'not integration'
```

The migrated reference simulation and its validation commands are documented in `docs/reference_pipeline.md`.
