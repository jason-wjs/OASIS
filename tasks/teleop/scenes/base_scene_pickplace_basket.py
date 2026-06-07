# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0     
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.teleop.tools import CameraBaseCfg
from tasks import ASSET_ROOT
from .scene_factory import MotionsSceneCfgFactory, RandomizableSceneMixin
@MotionsSceneCfgFactory.register("table_basket")
class TableBasketSceneCfg(InteractiveSceneCfg, RandomizableSceneMixin): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """

    # 机器人初始位姿 (MuJoCo [w,x,y,z] quaternion)
    robot_init_pos = (-2.46, -1.1, 0.76)
    robot_init_rot = (0.7071, 0, 0, -0.7071)

      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-2.5,-2.2, 0.755],
                                                rot=[0.5, 0.5, 0.5, 0.5]), 
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/FEZIBO_Standing_Desk_Black/table.usda",    # table model file
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )

    basket = RigidObjectCfg(
        prim_path="/World/envs/env_.*/basket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.75, -1.5, 0.05],
                                                  rot=[0, 0.7071, 0.7071, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/basket/basket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
        ),
    )

    cup = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.3, -2.0, 0.75],
                                                  rot=[0, 1.0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/objects/cup/cup.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
        ),
    )




    domelight = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0,)
    )

    def randomize_objects(self, scene, device):
        self._randomize_rigid_object(
            scene, device, "cup",
            x_range=(-0.1, 0.1), y_range=(0.0, 0.05),
        )

