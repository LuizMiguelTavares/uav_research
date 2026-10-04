"""Scene configuration for the validated ARL reference pipeline."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


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

    robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
