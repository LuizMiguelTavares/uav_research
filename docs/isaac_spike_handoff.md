# Isaac Sim / Isaac Lab research spike handoff

**Status:** Formally paused after accepting Isaac Sim / Isaac Lab as the working simulation stack. Resume substantial Isaac work only from a concrete experimental question.

## 1. Original objective

The spike tested whether Isaac Sim and Isaac Lab could serve as a practical simulation foundation for the UAV research program. The uncertainty was infrastructural: whether one local stack could support multirotor physics and control, native tensor access for learning, perception sensors, ROS 2 integration, coherent simulated time and transforms, visualization, and a path to reproducible experiments without modifying upstream Isaac Lab.

Success meant demonstrating an end-to-end reference pipeline, understanding its failure modes well enough to preserve correct behavior, and moving it into a clean external project that reproduces the result from its own environment. It did not mean selecting the final research problem.

The broader objective is a strong IROS 2027 submission by March 1, 2027 involving a quadrotor/UAV, reinforcement learning, perception, motion or trajectory planning, and meaningful real-hardware validation. The spike did not decide the final UAV, sensors, perception representation, RL method, MPC formulation, relationship between RL and MPC, moving-obstacle requirement, sim-to-real method, or scientific contribution.

## 2. Validated software and machine environment

| Component | Validated value |
|---|---|
| Operating system | Ubuntu 24.04.5 LTS |
| Python | 3.12.15 |
| Isaac Sim | 6.1 GA; wheel version `6.1.0.0` |
| Isaac Lab | 3.0.0-EA |
| Upstream Isaac Lab commit | `ae37b028ea415c91ea2bc32609efcd759ed2b974` |
| ROS | ROS 2 Jazzy |
| Physics | PhysX |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Python workflow | `uv`, with the `isaacsim` extra |

On Linux x86-64, the project dependency configuration selects the PyTorch CUDA 12.8 index. The spike did not require a separate project-managed CUDA toolkit decision.

The project is `~/research/uav_research`; its remote is `git@github.com:LuizMiguelTavares/uav_research.git` (`https://github.com/LuizMiguelTavares/uav_research.git`) and it is intended to remain private. It is an external Isaac Lab project. The upstream checkout remains a sibling, read-only dependency at `~/research/IsaacLab`, referenced through editable `uv` sources in `pyproject.toml`. Do not modify upstream to develop this project.

The migrated reference came from `IsaacLab/scripts/research_spike/validate_ros2_cmd_vel.py`; its recorded SHA-256 at migration was `462f7d397a3882c0c75d97eddfcd8ad114318465bdd1fb01fde5d97562c8319e`.

## 3. Current repository

```text
uav_research/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── reference_pipeline.md
│   ├── rtx_lidar_compatibility.md
│   └── isaac_spike_handoff.md
├── scripts/
│   ├── list_envs.py
│   └── standalone/
│       └── ros2_system_validation.py
├── src/uav_research/
│   ├── reference/
│   │   ├── scene_cfg.py
│   │   └── ros2_bridge.py
│   ├── sim/
│   │   └── rtx_lidar_accumulator.py
│   └── tasks/
│       └── __init__.py
└── tests/unit/
    └── test_rtx_lidar_accumulator.py
```

Responsibilities:

- `reference/scene_cfg.py` defines the current ARL Robot 1 system-validation fixture, ground and target geometry, Isaac Lab IMU, and depth camera.
- `reference/ros2_bridge.py` owns the reference ROS node, `/cmd_vel`, messages, camera calibration messages, validated point-cloud publication, and static/dynamic TF.
- `scripts/standalone/ros2_system_validation.py` is the executable integration fixture. It starts Kit with explicit requirements, constructs the scene and RTX LiDAR, runs control and simulation, schedules rendering and sensors, and reports LiDAR validation statistics.
- `sim/rtx_lidar_accumulator.py` is a deliberately profile-specific Isaac Sim 6.1 compatibility layer for trustworthy OS1 revolutions.
- `tasks/` is the Isaac Lab task entry-point area. It contains no registered scientific task yet.
- `tests/unit/` protects the accumulator contract without launching Kit.
- `docs/reference_pipeline.md` is the operational reference; `docs/rtx_lidar_compatibility.md` is the compatibility contract that must be read before changing RTX LiDAR behavior.

No planner, MPC, policy, perception-model, hardware-adapter, generic sensor, or generic vehicle package was created. Those packages would encode choices the research has not made and would add maintenance without enabling a current experiment.

## 4. Architectural principles

- Keep this as an external Isaac Lab project; do not patch the upstream checkout for project behavior.
- Use native Isaac Lab/PyTorch tensor paths for high-throughput RL observations, actions, rewards, randomization, and training. ROS is not a required transport inside training.
- Use ROS 2 at integration boundaries where messages, TF, visualization, external perception/control nodes, or hardware semantics are useful.
- Support both RL and MPC research. Evaluate them, when they exist, against common simulation scenarios, definitions, and metrics rather than building unrelated evaluation stacks.
- Keep real-hardware vendor drivers as external dependencies where practical. This repository should own research-specific configuration, adapters, semantics, and experiment integration.
- Do not create a generic hardware abstraction framework before concrete simulated and real systems expose a repeated interface problem.
- Treat current vehicles, sensors, controller, topics, scenes, and rates as reference fixtures. ARL Robot 1 and Ouster OS1 are not project-wide commitments.
- Never depend on persisted Isaac Sim or Kit user state for experiment-relevant behavior. Required runtime state belongs in version-controlled code or configuration and must be checked from a clean shell/profile.

## 5. Validated functionality

The spike demonstrated the following together in the reference fixture:

- ARL Robot 1 multirotor asset under PhysX;
- the Isaac Lab thruster actuator path;
- Lee velocity feedback control and rotor-thrust allocation;
- physics and controller execution at 100 Hz simulation cadence;
- a held ROS 2 `/cmd_vel` velocity/yaw-rate setpoint with a 0.5-second deadman timeout;
- ROS `/clock` from Isaac simulation time;
- consistent simulation-time stamps across custom ROS messages, TF, and RTX output;
- dynamic `world -> base_link` TF and static sensor transforms for `base_scan`, `imu_link`, and `front_depth_camera`;
- `/odom` and `/imu`;
- a front Isaac Lab depth camera, metric `32FC1` depth, `CameraInfo`, and a `mono8` debug image;
- conversion of depth plus `CameraInfo` to ROS `PointCloud2` through the standard Jazzy `depth_image_proc` node;
- RTX Ouster LiDAR raw output and manually validated 360-degree revolutions;
- publication of only validated LiDAR revolutions on `/point_cloud`;
- RViz with Fixed Frame `world`, showing the RTX LiDAR and depth-derived point cloud simultaneously.

The camera calibration is read from `depth_camera.data.intrinsic_matrices`, not recomputed from guessed focal pixels. The validated values are:

```ini
width = 320
height = 240
fx = 366.4996337890625
fy = 366.4996337890625
cx = 160.0
cy = 120.0
```

The simulated camera uses ideal/no distortion in `CameraInfo` (`plumb_bob`, zero coefficients). Raw depth is metres in `32FC1`, and image and calibration messages share the `front_depth_camera` frame and timestamp. The depth acquisition target is 20 Hz in simulation time.

The controller path was validated as an integration fixture, not as a final flight-control claim. In the current scene, zero velocity settles near ground contact; a positive vertical command was verified to produce upward motion.

## 6. Timing and loop structure

| Function | Simulation-time cadence |
|---|---:|
| Physics | 100 Hz |
| Lee feedback/control | 100 Hz |
| IMU | 100 Hz |
| Odometry | 50 Hz |
| RTX rendering | 40 Hz |
| Depth acquisition/publication | 20 Hz |
| Validated assembled LiDAR revolution | approximately 10 Hz |

The current loop deliberately separates rendering from depth acquisition: rendering at 40 Hz supports RTX LiDAR, while depth is requested only at its independent 20 Hz cadence.

The important order is:

1. compute and write the current actuator targets;
2. step physics with `sim.step(render=False)`;
3. refresh Isaac Lab scene state;
4. read Isaac simulation time and publish current dynamic TF;
5. render when the 40 Hz schedule is due, allowing RTX writers and GMO data to run;
6. validate and publish a newly completed LiDAR revolution;
7. publish odometry/IMU and acquire depth at their own cadences.

Publishing dynamic TF before the RTX render removed an arrival-order race: a cloud emitted during render must already have a transform at its exact simulation timestamp. Isaac simulation time is the single timestamp source, and RViz uses `use_sim_time=true`.

Configured simulation-time cadence and observed wall-clock topic rate are different. Rendering, processing load, and real-time factor can reduce wall-clock arrival rates without changing the simulated-time schedule.

## 7. RTX LiDAR investigation and preserved workaround

The installed native accumulator is not trusted for the validated profile. With:

```ini
config = "OS1"
variant = "OS1_REV6_32ch10hz512res"
tick_rate = 10.0
accumulate_outputs = True
```

RTX sometimes marked truncated data as `scanComplete=True`. Observed failures included structured missing angular sectors near the scan boundary and catastrophic outputs of exactly 32 points. Changing render cadence changed the missing wedge but did not eliminate it: 40 Hz was materially better than 20 Hz, while 100 Hz gave little further improvement. The failure was not an RViz, TF, self-occlusion, or normal Ouster firing-pattern explanation.

The preserved workaround is `RtxLidarRevolutionAccumulator`:

- set native `accumulate_outputs=False`;
- consume raw Generic Model Output segments;
- require ordered horizontal tick metadata for the 512-tick profile;
- recognize a boundary only from one verified `511 -> 0` wrap with `scanComplete=True`;
- discard the initial partial revolution and use its boundary only for synchronization;
- ignore duplicate GMO frame IDs;
- lose synchronization on a frame discontinuity, metadata-length mismatch, unexpected wrap, or ambiguous/missing boundary;
- reject incomplete or ambiguous revolutions instead of publishing them;
- assemble segments only between two verified boundaries;
- retain the latest valid native XYZ observation, original simulation timestamp, age, validity, and new-scan flag for possible future native RL use;
- publish assembled validated data on `/point_cloud` and leave `/point_cloud_raw` diagnostic-only;
- never restamp an old cloud after a rejected scan.

Completeness is not decided by a point-count threshold or by requiring a return in every azimuth bin; physically empty return regions are allowed. The exactly-32-point signature is covered by regression tests because it was an observed native failure, not because point count is the general validation rule.

This implementation is validated only for `OS1_REV6_32ch10hz512res` with 512 horizontal ticks and 32 emitters. It is a compatibility layer, not a universal LiDAR abstraction.

## 8. Clean-environment and Kit hidden-state failure

The original upstream spike unintentionally inherited Kit user state:

- `isaacsim.ros2.bridge` was persisted as enabled;
- `/persistent/UJITSO/geometry=false` was persisted.

A clean `uav_research` environment started Isaac Sim 6.1 but failed at `import isaacsim.core.experimental.utils.app` with `ModuleNotFoundError: isaacsim.core.experimental`. The upstream and clean environments had the same Isaac Sim/Isaac Lab package versions and byte-identical relevant extension payloads. Dependency resolution was not the cause. The experimental namespace appears when its Kit extension is enabled.

Enabling required extensions after `AppLauncher` started removed the Python import error but was too late to register `OmniSensorGenericLidarCoreAPI`. Pre-start registration was required. Differential testing also showed that clean-profile `BASIC` GMO metadata could report the BASIC auxiliary type while leaving `tickId` unfilled, making every revolution invalid.

The standalone fixture now supplies its requirements before Kit starts:

```text
--enable isaacsim.ros2.bridge
--/UJITSO/geometry=false
```

It also requests:

```ini
aux_output_level = "FULL"
```

`FULL` was repeatable in the final clean configuration and matches the installed RTX GMO examples/tests; `BASIC` was not sufficiently deterministic. Do not move these startup requirements to a post-launch extension call or reduce the auxiliary level without focused integration validation.

The general lesson is durable: simulation behavior relevant to scientific results must not depend on a developer's persisted Isaac/Kit profile. Extensions, renderer/sensor settings, and other startup dependencies must be explicit and version-controlled. Migration validation should include a clean shell/profile.

## 9. Final clean-environment validation

The final workflow used `uav_research/.venv`; it did not use the upstream IsaacLab Python environment or a `PYTHONPATH` workaround.

First 600-step run:

```ini
raw_segments = 240
raw_scans = 70
rejected = 4
valid = 66
points_min = 7547
points_mean = 7927.55
points_max = 8944
```

Fresh `bash --noprofile --norc` 600-step run:

```ini
raw_segments = 240
raw_scans = 70
rejected = 2
valid = 68
points_min = 7358
points_mean = 7864.69
points_max = 8094
```

Both runs produced populated tick metadata and validated revolutions. No 32-point scan reached the validated output. Afterward, the clean profile still had persisted geometry streaming enabled and no persisted enabled extensions, which demonstrated that the version-controlled startup arguments—not hidden profile state—made the run succeed.

## 10. Tests and quality checks

Final checkpoint results:

```yaml
pytest: 11 passed
ruff: passed
pre-commit: passed
```

The unit tests protect:

- startup synchronization without publishing the first partial revolution;
- verified `511 -> 0` boundaries;
- assembly across raw segments;
- duplicate frame suppression;
- frame discontinuity invalidation;
- missing, unexpected, and ambiguous boundaries;
- point/tick metadata-length mismatch;
- rejection of the observed 32-point complete-scan signature;
- spherical-to-Cartesian conversion;
- latest-valid observation timestamp, age, validity, and new-scan state.

These tests intentionally do not launch Kit. The bounded 600-step standalone command is the system/integration check for extension startup, schemas, rendering, sensors, control, ROS initialization, and real GMO metadata.

## 11. Important Git checkpoints

- `cfa9607 Initialize UAV research project` — initial external-project scaffold and locked environment.
- `d526403 Migrate validated ROS 2 reference pipeline` — behavior-preserving migration of the working spike.
- `1200a38 Protect validated RTX LiDAR accumulation` — extracted the accumulator and added its regression suite.
- `d59e029 Separate validated reference integration` — split scene and ROS integration from the standalone loop.
- `b3d99b1 Document validated research foundation` — added operational and compatibility documentation.
- `a3db4b1 Remove generated task assumptions` — removed scaffold choices that implied a scientific task prematurely.
- `c1db1ee Fix clean Isaac Sim startup requirements` — made Kit/RTX requirements explicit and reproduced the system in the project environment.
- `ea8f3c5 Document explicit Isaac runtime configuration` — made avoidance of persisted Kit state a durable repository rule.

The upstream Isaac Lab commit associated with this setup is `ae37b028ea415c91ea2bc32609efcd759ed2b974`.

## 12. Known warnings and limitations

- The project currently expects a sibling Isaac Lab checkout at `../IsaacLab` and uses editable local sources.
- The LiDAR accumulator is specific to one OS1 profile and its 512-tick boundary contract.
- `omni.timeline.get_timeline_interface().set_end_time(3600.0)` remains necessary. Before this fix, timeline wrap caused sustained RTX LiDAR output to stop while the rest of the simulation continued.
- The MotionBVH warning remains. It concerns missing motion effects/distortion in LiDAR data; it was not the scan-stopping or truncation cause.
- Fabric/renderer geometry-streaming warnings may still appear. The explicit startup configuration is the validated behavior.
- `FULL` GMO metadata has more overhead than `BASIC`.
- Native GMO bindings may print one initial `tickId is not filled` warning for an empty startup buffer; populated subsequent data is required before a revolution can validate.
- Wall-clock throughput can be below configured simulation-time sensor rates.
- A bounded run can finish before external ROS CLI discovery stabilizes; use the interactive reference run for live topic and TF validation.
- The reference zero-velocity behavior settles close to ground contact in the current scene.
- This is not yet a scientific Isaac Lab task. There is no final RL environment, MPC implementation, real-hardware pipeline, or sim-to-real experiment.
- The repository is private and intentionally has no project license yet.

## 13. Depth point-cloud performance observation

With the full visualization path active, including RViz and `depth_image_proc`, `/camera/depth/points` was observed at roughly **3–6 Hz wall-clock** even though depth acquisition is configured and validated at approximately **20 Hz simulation time**.

This is an uncharacterized performance observation. It does not show that the simulated depth sensor itself runs at only 3–6 Hz. Possible contributors include simulation real-time factor, RTX rendering, message conversion, `depth_image_proc`, RViz, or their combined load. The spike deliberately stopped before profiling them.

## 14. When performance profiling should resume

Generic profiling is deferred until an experiment creates a decision-relevant constraint. Legitimate triggers include:

- a policy actually consumes depth or LiDAR;
- sensor frequency becomes an independent variable or requirement;
- many parallel visual environments are required;
- perception becomes the training-throughput bottleneck;
- near-real-time execution becomes necessary;
- sim-to-real execution rate becomes relevant.

At that point, profile the exact task, observation pipeline, environment count, and evaluation requirement. Do not optimize the reference visualization stack in isolation.

## 15. Reference fixtures versus paper decisions

These are current validation fixtures, not final research choices:

- ARL Robot 1;
- Ouster OS1 Rev6, 32 channels, 10 Hz, 512 horizontal ticks;
- the current front depth camera and its calibration;
- Lee velocity control and current gains;
- current ROS topic names and frame layout;
- the ground, target cuboid, walls, or obstacles used during validation;
- 40 Hz RTX rendering;
- the current exact sensor mounts and arrangement.

A future agent must not infer that these define the IROS paper. They may be reused temporarily when doing so reduces setup work without prejudging the research question.

## 16. Open scientific questions

The following remain deliberately open:

- final UAV platform;
- final sensor suite;
- perception representation;
- RL algorithm;
- observation and action spaces;
- motion/trajectory-planning architecture;
- moving-obstacle requirement;
- perception-aware problem formulation;
- domain randomization and sim-to-real strategy;
- MPC formulation;
- RL/MPC comparison, division of responsibility, or integration;
- real-hardware software architecture;
- final training and evaluation scenarios, baselines, and metrics;
- the final scientific contribution.

These should be decided from research hypotheses and experimental evidence, not from the reference fixture.

## Post-spike opportunity: Fly4Future X500 / CTU-MRS lineage

We have access to a physical Fly4Future X500 UAV. [Fly4Future's official X500 description](https://fly4future.com/custom-drones/x500-darpa-research-drone/) states that the X500 was influenced by the CTU Multi-Robot Systems Group's DARPA SubT platform and that the platform described by Petráček et al. in *Large-Scale Exploration of Cave Environments by Unmanned Aerial Vehicles* served as a foundation for X500 development. This confirms lineage, not exact hardware or sensor equivalence between our purchased vehicle and the paper's UAV. The actual vehicle configuration must be audited before reuse.

When work resumes, existing CTU-MRS and Fly4Future resources may provide a better starting point for an Isaac-native representation than modelling the vehicle from scratch. Potentially useful material includes geometry and meshes, mass and inertia, propulsion configuration, sensor extrinsics, physical limits, and simulator parameters. Every imported parameter must record its provenance as measured/identified, manufacturer or design nominal, geometry-derived, simulator-tuned, assumed/default, or unknown.

Reusing CTU-MRS physical/model resources and integrating the MRS UAV System are separate opportunities. The MRS control, estimation, and planning stack should remain optional and external unless a future research question justifies integration. A robust Isaac/MRS integration could eventually be useful upstream, but it is not a current objective. This opportunity does not reopen the spike; the Isaac Sim / Isaac Lab Research Spike remains formally paused.

## 17. Work paused as useful but not timely

Do not continue these topics without a concrete experimental need:

- deeper UJITSO internals;
- replacing the working RTX workaround for elegance;
- optimizing `BASIC` versus `FULL` GMO metadata;
- generic RViz/`depth_image_proc` performance profiling;
- exact OS1-16 fidelity;
- MotionBVH optimization;
- headless RTX investigation;
- photorealism or cinematic rendering;
- PX4, Nav2, Gazebo, MAVROS, or similar integration without an experiment that requires it;
- generic hardware abstraction layers;
- permanent multi-agent infrastructure for the repository;
- extensive Isaac architecture refactoring.

## 18. Why the spike is complete

Isaac Sim / Isaac Lab is accepted as the working simulation stack for this research program. The spike demonstrated the required control, sensor, time, ROS, visualization, and project-reproducibility foundation and documented the compatibility behavior that was expensive to discover.

The remaining uncertainty is predominantly scientific rather than infrastructural. It will be reduced more effectively by defining and running concrete tasks than by broader Isaac exploration. The next substantial Isaac work should begin with an experimental question, measurable outcome, and reproduction plan.

## 19. Essential commands

Set up a fresh shell and project environment:

```bash
cd ~/research/uav_research
source /opt/ros/jazzy/setup.bash
uv sync --extra isaacsim
```

Run unit tests and project checks:

```bash
uv run pytest tests/unit
uv run ruff check .
uv run pre-commit run --all-files
```

Run the interactive reference fixture:

```bash
uv run --extra isaacsim python scripts/standalone/ros2_system_validation.py \
  --visualizer kit
```

Run the bounded integration check:

```bash
uv run --extra isaacsim python scripts/standalone/ros2_system_validation.py \
  --visualizer kit --max_steps 600
```

In another ROS-sourced terminal, generate the depth point cloud:

```bash
ros2 run depth_image_proc point_cloud_xyz_node --ros-args \
  -p use_sim_time:=true \
  -r image_rect:=/camera/depth/image_raw \
  -r camera_info:=/camera/depth/camera_info \
  -r points:=/camera/depth/points
```

For RViz, use `use_sim_time=true`, Fixed Frame `world`, RTX PointCloud2 topic `/point_cloud`, and depth PointCloud2 topic `/camera/depth/points`. Useful live checks are maintained in `docs/reference_pipeline.md`.

## 20. Next smallest experiment when Isaac work resumes

Implement one minimal manager-based UAV navigation/control task to prove the full scientific workflow. It may use ARL Robot 1 as a temporary fixture and should use native Isaac Lab/PyTorch observations rather than ROS transport. Give it one measurable objective, a small deterministic evaluation set, explicit metrics, a fixed training configuration, and a reproducible checkpoint/evaluation command.

A high-fidelity X500 model is not required for the first scientific workflow task. Do not block initial RL/MPC experiments on CTU-MRS model integration.

The success criterion is the workflow:

```text
task definition -> training -> evaluation -> metrics -> reproducibility
```

This experiment is a workflow proof, not the selected IROS problem. Its purpose is to expose the next decision-relevant research and infrastructure questions with minimal new surface area.
