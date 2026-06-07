import numpy as np
import torch
from tasks.teleop.g1_config.g1 import G1_CYLINDER_CFG
from tasks.teleop.tools import *
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz


class RandomizableSceneMixin:
    """场景级随机化基类。

    子类按需重写 ``randomize_objects``；常见的"单物体 box 偏移 + yaw"用例可以
    直接复用 ``_randomize_rigid_object`` 帮助函数。
    """

    def randomize_objects(self, scene, device):
        """默认不随机化任何物体；子类按需重写。"""
        pass

    def _randomize_rigid_object(
        self,
        scene,
        device,
        name,
        x_range=(0.0, 0.0),
        y_range=(0.0, 0.0),
        z_range=(0.0, 0.0),
        yaw_range=(-np.pi, np.pi),
    ):
        """对 ``scene[name]`` 在 default_root_state 基础上叠加随机偏移与 yaw。

        ``x/y/z_range`` 给定 (low, high) 区间，``yaw_range=None`` 表示不随机朝向。
        """
        asset = scene[name]
        state = asset.data.default_root_state.clone()
        state[:, 0] += np.random.uniform(*x_range)
        state[:, 1] += np.random.uniform(*y_range)
        state[:, 2] += np.random.uniform(*z_range)
        if yaw_range is not None:
            roll, pitch, _ = euler_xyz_from_quat(state[:, 3:7])
            yaw = torch.tensor([np.random.uniform(*yaw_range)], device=device)
            state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        asset.write_root_state_to_sim(state)

class MotionsSceneCfgFactory:
    """Motions场景配置工厂类 - 带注册功能"""

    # 类级别的注册表
    _scene_registry = {}

    @classmethod
    def register(cls, scene_type: str):
        """注册装饰器"""

        def decorator(scene_class):
            cls._scene_registry[scene_type] = scene_class
            print(f"Registered scene type: {scene_type} -> {scene_class.__name__}")
            return scene_class

        return decorator

    @classmethod
    def create(cls, scene_type: str = "simple_room", **kwargs):
        """创建场景配置实例

        Scenes may declare class-level ``robot_init_pos`` / ``robot_init_rot``
        to override the default G1 spawn pose (位置 / 朝向, MuJoCo `[w,x,y,z]`
        quaternion).
        """
        if scene_type not in cls._scene_registry:
            available = list(cls._scene_registry.keys())
            raise ValueError(f"未知场景类型: {scene_type}。可用: {available}")

        BaseClass = cls._scene_registry[scene_type]

        init_pos = getattr(BaseClass, "robot_init_pos", None)
        init_rot = getattr(BaseClass, "robot_init_rot", None)
        if init_pos is not None or init_rot is not None:
            new_init_state = G1_CYLINDER_CFG.init_state.replace(
                pos=tuple(init_pos) if init_pos is not None
                    else G1_CYLINDER_CFG.init_state.pos,
                rot=tuple(init_rot) if init_rot is not None
                    else G1_CYLINDER_CFG.init_state.rot,
            )
            robot_cfg = G1_CYLINDER_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
                init_state=new_init_state,
            )
        else:
            robot_cfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        @configclass
        class MotionsSceneCfg(BaseClass):
            robot: ArticulationCfg = robot_cfg
            front_camera = CameraPresets.g1_front_camera()
            left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
            right_wrist_camera = CameraPresets.right_dex3_wrist_camera()

        MotionsSceneCfg.__name__ = f"Motions{BaseClass.__name__}"
        instance = MotionsSceneCfg(**kwargs)
        instance.__dict__.pop("robot_init_pos", None)
        instance.__dict__.pop("robot_init_rot", None)
        return instance

    @classmethod
    def list_registered(cls):
        """列出所有已注册的场景"""
        return dict(cls._scene_registry)