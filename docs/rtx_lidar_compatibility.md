# Isaac Sim 6.1 RTX LiDAR accumulation compatibility

## Scope

This document applies to the experimentally validated configuration:

- `config="OS1"`
- `variant="OS1_REV6_32ch10hz512res"`
- `tick_rate=10.0`
- 512 horizontal ticks and 32 emitters
- Isaac Sim 6.1 GA / Isaac Lab 3.0.0-EA

It is not a universal LiDAR accumulator design.

## Confirmed native-accumulator failure

With native `accumulate_outputs=True`, RTX sometimes returned truncated revolutions while reporting `scanComplete=True`. Failures included structured wedges near the scan boundary and catastrophic 32-point outputs. Changing rendering from 20 to 40 or 100 Hz changed the artifact but did not eliminate it.

## Workaround contract

`uav_research.sim.rtx_lidar_accumulator.RtxLidarRevolutionAccumulator` operates on raw GMO segments with native accumulation disabled. It:

1. ignores duplicate GMO frame IDs;
2. invalidates the current revolution when frame IDs are discontinuous;
3. accepts a boundary only when `scanComplete=True`, exactly one ordered wrap occurs, and the wrap is `511 -> 0`;
4. rejects scan-complete data with missing or ambiguous boundaries;
5. rejects unexpected wraps when `scanComplete=False`;
6. uses the first verified boundary only to synchronize startup;
7. assembles points between two verified boundaries;
8. retains the latest valid native XYZ observation, its original simulation timestamp, age, validity, and new-scan state;
9. leaves ROS `PointCloud2` construction to the reference ROS bridge.

No point-count threshold or requirement that every azimuth bin contain a return is used. Empty return regions can be physically valid.

## Required Kit startup configuration

The standalone reference script explicitly enables `isaacsim.ros2.bridge` and disables UJITSO geometry streaming before `AppLauncher` starts Kit. The bridge dependency registers the RTX sensor schemas and the experimental Isaac Sim Python namespace. In a fresh Isaac Sim 6.1 profile with geometry streaming enabled, GMO can report `AuxType.BASIC` while leaving `tickId` unfilled, so every revolution must be rejected. The script therefore starts Kit with `--/UJITSO/geometry=false` and requests `aux_output_level="FULL"`, matching the metadata level used by the installed RTX GMO examples and tests.

Do not move the bridge or geometry setting to a post-start call. RTX schemas and geometry-streaming mode are initialized during Kit startup. Do not reduce the auxiliary level without rerunning the bounded integration check below.

## ROS publication

- `/point_cloud_raw` is the optional native writer output for diagnostics.
- `/point_cloud` contains only manually assembled validated revolutions.
- A rejected revolution is dropped. An old cloud is not restamped because the vehicle may have moved.
- Validated scan timestamps use the Isaac simulation time at the render that completes the revolution, matching the installed RTX ROS writer semantics.

## Regression protection

The unit suite covers startup synchronization, verified wraps, duplicate frames, frame discontinuities, ambiguous boundaries, mismatched metadata lengths, the observed 32-point failure signature, spherical-to-Cartesian conversion, and native observation freshness.

```bash
uv run pytest tests/unit
```

These tests do not launch Isaac Sim. The bounded reference run is the RTX integration check:

```bash
uv run --extra isaacsim python scripts/standalone/ros2_system_validation.py \
  --visualizer kit --max_steps 600
```

A successful run must publish no 32-point validated cloud and must report normal full-scan point counts after warm-up. The migration validation produced 65 valid revolutions from 600 physics steps, with 7,403–8,268 points including warm-up and no 32-point validated output.
