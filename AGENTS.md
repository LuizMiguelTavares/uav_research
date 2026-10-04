# UAV Research Agent Guide

- Use the exact locally installed Isaac Lab and Isaac Sim source as the authority for version-specific APIs.
- Treat `../IsaacLab` as an external read-only dependency. Do not modify, clean, reset, or commit it from this repository.
- This repository supports shared UAV RL and MPC research. The final vehicle, sensors, algorithms, and sim-to-real strategy remain open.
- Keep high-throughput training and native simulation algorithms on Isaac Lab/PyTorch data paths. Use ROS where message, TF, visualization, integration, or hardware semantics provide value.
- Preserve the validated simulation-time, `/clock`, timestamp, and TF ordering documented in `docs/reference_pipeline.md`.
- Preserve the RTX LiDAR compatibility behavior documented in `docs/rtx_lidar_compatibility.md`. Do not replace it with native accumulation or point-count heuristics without focused evidence.
- Validate sensor, rendering, controller-rate, or timing changes with narrow experiments before combining them.
- Avoid speculative factories, hardware abstraction layers, empty algorithm packages, and unrelated refactors.
- Prefer reproducible scientific progress over infrastructure polish.
