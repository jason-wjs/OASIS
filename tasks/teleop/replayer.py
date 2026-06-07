import torch
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
import omni
import random
import math
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from tools import *
from scenes import MotionsSceneCfgFactory
from tasks.params import ASSET_ROOT
                    
@configclass
class ObservationsCfg:
    pass

@configclass
class ActionsCfg:
    pass

@configclass
class EventCfg:
    randomize_background_textures = EventTerm(
        func=randomize_texture,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("room_walls"),
            "texture_groups": { 
                "Satin_Paint_Gray": [
                    str(ASSET_ROOT / "textures/background_textures/*.jpg"),
                    str(ASSET_ROOT / "textures/vMaterials_2/Concrete/textures/*diff.jpg"),
                ],
                "Concrete_Polished":[
                    str(ASSET_ROOT / "textures/vMaterials_2/Concrete/textures/*diff.jpg"),
                    str(ASSET_ROOT / "textures/floor_textures/*.jpg"),
                ]
            },
            "event_name": "texture_randomizer",
            "material_names": ["*"],
            "verbose": True  # 打印当前asset下所有绑定的material及其纹理/mesh列表，方便配置texture_groups
        },
    )

    randomize_table = EventTerm(
        func=randomize_texture,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("packing_table"),
            "texture_groups": { 
                "TableTop": [
                     str(ASSET_ROOT / "mdl/vMaterials_2/Wood/textures/*diff.jpg"),
                     str(ASSET_ROOT / "textures/table/*.jpg"),
                     str(ASSET_ROOT / "textures/vMaterials_2/Wood/textures/*diff.jpg"),
                ]
            },
            "event_name": "table_randomizer",
            "material_names": ["*"],
            "verbose": True  # 打印当前asset下所有绑定的material及其纹理/mesh列表，方便配置texture_groups
        },
    )

    randomize_light = EventTerm(
        func=randomize_light, 
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("room_walls"),
            "domelight_cfg": SceneEntityCfg("domelight"),
        },
    )

    # randomize_robot_texture = EventTerm(
    #     func=randomize_robot_texture,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #     },
    # )

    randomize_cameras = EventTerm(
        func=randomize_cameras,
        mode="reset",
        params={
            "camera_specs": {
                "front_camera": {
                    "pos_noise": (0.01, 0.01, 0.01),
                    "rot_noise_deg": 1.5,
                },
                "left_wrist_camera": {
                    "pos_noise": (0.01, 0.01, 0.01),
                    "rot_noise_deg": 1.5,
                },
                "right_wrist_camera": {
                    "pos_noise": (0.01, 0.01, 0.01),
                    "rot_noise_deg": 1.5,
                },
            },
            "verbose": True,
        },
    )
    
class Replayer:
    def __init__(self,
                 input_dir,
                 record_dir,
                 num_envs=1,
                 target_envs_per_episode=1,
                 device='cuda',
                 start=None,
                 end=None):

        self.input_dir = Path(input_dir)
        all_files = sorted(self.input_dir.glob("**/data.json"))
        if not all_files:
            raise FileNotFoundError(f"在 {input_dir} 中没有找到任何 JSON 文件")

        self.json_files = []
        for f in all_files:
            num = int(f.parent.name.split("_")[-1])
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue
            self.json_files.append(f)


        if not self.json_files:
            raise ValueError(f"没有匹配的 episode (start={start}, end={end})")

        print(f"回放 {len(self.json_files)}/{len(all_files)} 个 episode")

        self.record_dir = record_dir
        self.num_envs = num_envs
        self.target_envs_per_episode = target_envs_per_episode
        self.device = device
        self._running = True

        _, first_scene = load_and_save_robot_data(str(self.json_files[0]))
        self.scene_name = first_scene
        
        # 初始化环境配置
        scene_cfg = MotionsSceneCfgFactory.create(scene_type=self.scene_name, num_envs=self.num_envs, env_spacing=20.0)
        scene_cfg.replicate_physics = False

        env_cfg = ManagerBasedEnvCfg()
        env_cfg.dt = 0.005
        env_cfg.scene = scene_cfg
        env_cfg.events = EventCfg()
        env_cfg.decimation = 4  
        env_cfg.observations = ObservationsCfg()
        env_cfg.actions = ActionsCfg()
        
        self.env = ManagerBasedEnv(cfg=env_cfg)
        self.env.sim.set_setting("/rtx/rendermode", "PathTracing") # 开启路径追踪渲染
        self.env.sim.set_setting("/rtx/pathtracing/optixDenoiser/enabled", True) # 开启 OptiX 降噪器
        self.env.sim.set_setting("/rtx/pathtracing/totalSpp", 32) # 设置路径追踪采样数
        self.env.sim.set_setting("/rtx/post/motionblur/enabled", True) 

        self.env_origins = self.env.scene.env_origins.clone()
        
        # 初始化记录器
        self.recorders = []
        for i in range(self.num_envs):
            env_record_dir = record_dir
            self.recorders.append(StateRecorder(env_record_dir, skip_json=True))

    def broadcast_tensors_recursive(self, data, num_envs):
        if isinstance(data, torch.Tensor):
            if data.shape[0] == 1:
                return data.repeat(num_envs, *([1] * (data.dim() - 1)))
            return data
        elif isinstance(data, dict):
            return {k: self.broadcast_tensors_recursive(v, num_envs) for k, v in data.items()}
        return data


    def _count_completed_envs(self, epi_root_dir):
        """以 .done 标记为准统计已完成的 env 数量，并清理残缺目录（上次中断遗留）。"""
        completed = 0
        for d in sorted(os.listdir(epi_root_dir)):
            if not d.startswith("env_"):
                continue
            p = os.path.join(epi_root_dir, d)
            if not os.path.isdir(p):
                continue
            if os.path.exists(os.path.join(p, ".done")):
                completed += 1
            else:
                print(f"  [cleanup] 移除残缺目录: {d}")
                shutil.rmtree(p, ignore_errors=True)
        return completed

    def replay(self):
        """遍历所有 JSON 文件并执行回放，支持断点续跑 + 每 episode 目标 env 数。"""
        try:
            for json_file in self.json_files:
                if not self._running: break

                epi_name = json_file.parent.name
                epi_root_dir = os.path.join(self.record_dir, epi_name)
                os.makedirs(epi_root_dir, exist_ok=True)

                saved = self._count_completed_envs(epi_root_dir)
                if saved >= self.target_envs_per_episode:
                    print(f"\n[skip] {epi_name} 已完成 {saved}/{self.target_envs_per_episode}")
                    continue

                print(f"\n正在处理文件: {epi_name} (已完成 {saved}/{self.target_envs_per_episode})")

                sim_state_list, _ = load_and_save_robot_data(str(json_file), epi_root_dir)
                num_steps = len(sim_state_list)

                env_ids = torch.arange(self.num_envs, device=self.device)

                while saved < self.target_envs_per_episode and self._running:
                    this_round = self.num_envs

                    self.env.reset()  # 触发 EventCfg 里的所有 randomize_*

                    save_dirs = []
                    for k in range(this_round):
                        save_dir = os.path.join(epi_root_dir, f"env_{saved + k:05d}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_dirs.append(save_dir)
                        self.recorders[k].create_episode(scene=self.scene_name, save_dir=save_dir)

                    round_ok = True
                    for t in tqdm(range(num_steps),
                                desc=f"{epi_name} [{saved}/{self.target_envs_per_episode}] round n={this_round}"):
                        if not self._running:
                            round_ok = False
                            break
                        broadcasted_state = self.broadcast_tensors_recursive(sim_state_list[t], self.num_envs)
                        self.env.scene.reset_to(broadcasted_state, env_ids, is_relative=True)
                        self.env.sim.render()

                        # replay 模式下 dt 被忽略：get_camera_image 内部改用每个相机自己的 update_period 驱动，
                        # 自动适配 camera_config 里的 update_period，保证每帧都能拿到新图。
                        images = get_camera_image(self.env.scene, 0.0, replay_mode=True, num_envs=self.num_envs)
                        for k in range(this_round):
                            self.recorders[k].save_data(
                                replay_mode=True,
                                camera_images=images[k],
                                camera_order=['head', 'left', 'right'],
                            )

                    if not round_ok:
                        break

                    for k in range(this_round):
                        self.recorders[k].save_episode()
                        Path(save_dirs[k], ".done").touch()

                    saved += this_round


        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()

        finally:
            try:
                for env_idx in range(self.num_envs):
                    self.recorders[env_idx].close()
                print("Recorder saved.")
            except:
                pass

            stop_async_writer()
            os._exit(0)

    def _signal_handler(self, signum, frame):
        print("\n[Termination] Shutdown signal received. Cleaning up...")
        self._running = False