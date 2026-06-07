import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.math import quat_apply, quat_mul
from tasks import ASSET_ROOT
from .scene_factory import MotionsSceneCfgFactory, RandomizableSceneMixin

@MotionsSceneCfgFactory.register("wipe_screen")
class WipeScreenSceneCfg(InteractiveSceneCfg, RandomizableSceneMixin):

    robot_init_pos = (-2.46, -1.6, 0.76)
    robot_init_rot = (0.7071, 0, 0, -0.7071)

    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # 房间中心
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2.5, -2.2, 0.755],  # TODO: 待坐姿姿态下目测微调高度/距离
            rot=[0.5, 0.5, 0.5, 0.5],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/FEZIBO_Standing_Desk_Black/table.usda",
        ),
    )


    sponge = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sponge",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.7, -2.0, 0.8],
                                                  rot=[0, 0.7071068, 0.7071068, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/sponge/sponge.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
        ),
    )

    screen = RigidObjectCfg(
        prim_path="/World/envs/env_.*/screen",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.5, -2.15, 0.76],
                                                  rot=[0.0, 0.0, 1.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/screen/screen.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
        ),
    )

    domelight = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    def randomize_objects(self, scene, device):
        pass
        # self._randomize_rigid_object(
        #     scene, device, "sponge",
        #     x_range=(-0.05, 0.15), y_range=(0.0, 0.05), yaw_range=None
        # )
