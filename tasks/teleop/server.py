import argparse
import json
import logging
import time
import numpy as np
import redis
import torch
from rich import print
from typing import Type
import os
import signal
from functools import partial
from tasks import isaaclab_g1_hand_joint_names, mujoco_g1_hand_joint_names, REDIS_ACTION_KEY
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, help='Path to Controller ONNX policy file')
parser.add_argument('--scene', type=str, default="simple_room",
                    help='Scene cfg name')
parser.add_argument('--controller', type=str, required=True, choices=["teleopit"],
                    help='RL controller name')
parser.add_argument('--robot', type=str, default="unitree_g1", help='Robot name')
parser.add_argument('--dt', type=float, required=True, help='Simulation timestep')
parser.add_argument('--decimation', type=int, required=True, help='Number of simulation steps per control step')

# Redis related arguments
parser.add_argument('--redis_ip', type=str, default='localhost', help='Redis host for the action stream')
parser.add_argument('--redis_port', type=int, default=6379, help='Redis port')
parser.add_argument('--redis_db', type=int, default=0, help='Redis db index')

# Record related arguments
parser.add_argument('--record', action='store_true',
                    help='Record data to json')
parser.add_argument("--text", type=str, default=None,
                    help="task text instruction; written into data.json text field when --record is on")
parser.add_argument('--record_dir', type=str, help='record dir ')

# Replay related arguments
parser.add_argument('--replay', action='store_true',
                    help='Replay mode, need json file')
parser.add_argument('--input_dir', type=str, help='replay input dir')
parser.add_argument('--output_dir', type=str, help='aug output dir ')
parser.add_argument('--num_envs', type=int, default=1,
                    help='replay 时单轮并行渲染的 env 数 (受显存限制)')
parser.add_argument('--target_envs_per_episode', type=int, default=1,
                    help='replay 时每个 episode 目标累计渲染的 env 数，支持跨轮累积与断点续跑')
parser.add_argument('--start', type=int, default=None,
                    help='replay 起始 episode 编号 (含)，如 episode_0005 即 5')
parser.add_argument('--end', type=int, default=None,
                    help='replay 结束 episode 编号 (含)')

# image server related arguments
parser.add_argument("--low_render", action="store_true", 
                    help="disable some render options for camera frames")
parser.add_argument("--camera_jpeg", action="store_true",
                    help="enable JPEG compression for camera frames")
parser.add_argument("--camera_jpeg_quality", type=int, default=85, help="JPEG quality (1-100)")
parser.add_argument("--skip_cvtcolor", action="store_true", default=False,
                    help="skip cv2.cvtColor if upstream already BGR")

parser.add_argument("--pico_host", type=str, default=None,
                    help="PICO headset IP for live front_camera preview (no streaming if not set)")
parser.add_argument("--pico_port", type=int, default=12345,
                    help="PICO XRoboToolkit TCP port (must match the APK's video_source.yml)")
parser.add_argument("--pico_fps", type=int, default=50,
                    help="PICO H.264 encode fps; should match front_camera tick rate")
parser.add_argument("--pico_bitrate", type=int, default=4_000_000,
                    help="PICO H.264 bitrate in bps; default 4Mbps matches the ZEDMINI APK profile")


AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from tools import *
from scenes import MotionsSceneCfgFactory
from controller import *
from tasks.teleop.replayer import Replayer
from tasks.params import DEFAULT_MIMIC_OBS
isaaclab_to_mujoco_reindex_hand = [isaaclab_g1_hand_joint_names.index(name) for name in mujoco_g1_hand_joint_names]
mujoco_to_isaaclab_reindex_hand = [mujoco_g1_hand_joint_names.index(name) for name in isaaclab_g1_hand_joint_names]


class RealTimePolicyController:
    def __init__(self,
                 sim: sim_utils.SimulationContext,
                 scene: InteractiveScene,
                 policy_path,
                 controller: Type[BaseController],
                 device='cuda',
                 record=False,
                 record_dir=None,
                 robot_name="unitree_g1",
                 dt=0.005,
                 decimation=4,
                 redis_host='localhost',
                 redis_port=6379,
                 redis_db=0,
                 ):
        self.redis_client = None
        self.redis_pipeline = None
        self.redis_key = REDIS_ACTION_KEY
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            self.redis_pipeline = self.redis_client.pipeline()
            self.redis_pipeline.delete(self.redis_key)
            self.redis_pipeline.execute()
        except Exception as e:
            print(f"Error connecting to Redis at {redis_host}:{redis_port}/{redis_db}: {e}")

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)
        # Create Isaacsim sim
        self.scene = scene
        self.robot = self.scene["robot"]
        # body_names 在运行期固定，索引只需查一次（teleopit anchor_body_name = torso_link）
        self.torso_link_idx = list(self.robot.data.body_names).index("torso_link")
        self.sim = sim
        self.dt = dt
        self.decimation = decimation
        self.sim_duration = 100000.0
        self.robot_name = robot_name

        if args.low_render:
            self.sim.set_setting("/rtx/rendermode", "RealTime") 
            self.sim.set_setting("/rtx/post/aa/op", 0)  # 关闭抗锯齿 (DLSS/FXAA)
            self.sim.set_setting("/rtx/reflections/enabled", False) # 关闭反射
            self.sim.set_setting("/rtx/shadows/enabled", False)     # 关闭阴影
            self.sim.set_setting("/rtx/directLighting/enabled", True) # 仅保留基础直接光
            self.sim.set_setting("/rtx/indirectLighting/enabled", False) # 关闭间接光

        self.controller = controller(policy_path, device)
        
        self.record = record
        if record and record_dir is None:
            logging.warning("Record dir is not set! Data won't record.")
            self.record = False

        self.recorder = StateRecorder(record_dir, text=args.text) if self.record else None

        self._running = True

        # 捕获 Ctrl+C / kill：置标志让主循环退出，走 cleanup() 做优雅关闭
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[Termination] Shutdown signal received. Cleaning up...")
        self._running = False

    def reset(self):
        """Reset robot to initial position (IsaacLab)"""
        self.robot.reset()
        self.sim.reset()

        randomize = getattr(self.scene.cfg, "randomize_objects", None)
        if randomize is not None:
            randomize(self.scene, self.device)

        self.controller.reset()
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_state_to_sim(root_state)

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.dt)

    def extract_data(self):
        """Extract robot state data"""

        dof_pos = self.robot.data.joint_pos[0, :].clone()
        dof_vel = self.robot.data.joint_vel[0, :].clone()

        # --- Base ---
        root_pos = self.robot.data.root_pos_w[0, :].clone()
        quat = self.robot.data.root_quat_w[0, :].clone()  # (w, x, y, z)
        lin_vel = self.robot.data.root_lin_vel_b[0, :].clone()
        ang_vel = self.robot.data.root_ang_vel_b[0, :].clone()

        # --- Torso anchor quat (teleopit anchor_body_name = torso_link) ---
        body_quat_attr = self.robot.data.body_link_quat_w
        torso_quat = body_quat_attr[0, self.torso_link_idx, :].clone()

        root_pos = root_pos.detach().cpu().numpy()
        dof_pos = dof_pos.detach().cpu().numpy()
        dof_vel = dof_vel.detach().cpu().numpy()
        quat = quat.detach().cpu().numpy()
        lin_vel = lin_vel.detach().cpu().numpy()
        ang_vel = ang_vel.detach().cpu().numpy()
        torso_quat = torso_quat.detach().cpu().numpy()

        return [root_pos, quat, lin_vel, ang_vel,
                dof_pos[isaaclab_to_mujoco_reindex_hand],
                dof_vel[isaaclab_to_mujoco_reindex_hand],
                torso_quat]


    def run(self):
        """Main simulation loop"""
        print("Starting simulation...")
        self.reset()

        steps = int(self.sim_duration / self.dt)

        button_a_pressed = None
        button_b_pressed = None

        last_policy_time = None

        target_policy_period = self.dt * self.decimation  # 0.02s @ 50Hz
        next_policy_deadline = None

        start_receive_ref_data = False
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 500

        recording = False
        saving = False
        pd_target_tensor = self.robot.data.default_joint_pos.clone()

        last_valid_data = None
        try:
            for i in range(steps):
                if not self._running:
                    break
                self.sim.step(render=False)
                self.scene.update(dt=self.dt)
                
                if i % self.decimation == 0:
                    self.sim.render()
                    # 图像写进共享内存
                    get_camera_image(self.scene, self.dt * self.decimation)
                    proprio_data = self.extract_data()
                    
                    self.redis_pipeline.get(self.redis_key)
                    redis_results = self.redis_pipeline.execute()
                    last_valid_data = json.loads(redis_results[0]) if redis_results[0] else None

                    if last_valid_data is None:
                        if not start_receive_ref_data and i % (self.decimation * 50) == 0:
                            print("Waiting for first data pack...")
                        action_mimic = {k: (v.copy() if hasattr(v, "copy") else list(v)) for k, v in DEFAULT_MIMIC_OBS[self.robot_name].items()} 
                        action_mimic["root_pos"] = self.robot.data.default_root_state.clone()[0, :3].detach().cpu().numpy() # 用sim里最新的root位置覆盖mimic obs里的root位置，保证起步时动作和sim里状态一致，避免一开始就大跳
                        action_mimic["root_quat"] = self.robot.data.default_root_state.clone()[0, 3:7].detach().cpu().numpy() # 同上，保持起步时姿态一致
                        action_left_hand = np.zeros(7)
                        action_right_hand = np.zeros(7)
                    else:
                        if not start_receive_ref_data:
                            print("Received first data pack, starting control loop.")
                            start_receive_ref_data = True
                        action_mimic = last_valid_data["body"]
                        action_left_hand = last_valid_data["left_hand"]
                        action_right_hand = last_valid_data["right_hand"]
                        reset = last_valid_data.get("reset_robot", False)
                        button_a_pressed = last_valid_data.get('start_record', False)
                        button_b_pressed = last_valid_data.get('start_save', False)
                        if reset:
                            if recording: 
                                self.recorder.clear()
                            print("Resetting robot and clearing data buffer.")
                            last_valid_data = None
                            recording = False
                            saving = False
                            start_receive_ref_data = False
                            self.reset()
                            continue
                        
                    compute_start = time.time()
                    pd_target = self.controller.step(proprio_data, action_mimic, start_receive_ref_data)
                    compute_elapsed = time.time() - compute_start

                    left_hand_q = action_left_hand
                    right_hand_q = action_right_hand
                    pd_target = np.concatenate([pd_target[:22], left_hand_q, pd_target[22:], right_hand_q])
                

                    # PD control
                    pd_target_tensor = torch.from_numpy(pd_target[mujoco_to_isaaclab_reindex_hand]).to(
                        self.device, dtype=torch.float32
                    )
                    self.robot.set_joint_position_target(pd_target_tensor)
                    self.scene.write_data_to_sim()

                    if self.recorder is not None:
                        if button_a_pressed and self.recorder.is_available() and not recording:
                            if self.recorder.create_episode(scene=args.scene):
                                print("Start recording episode.")
                                recording = True
                                
                        if recording:
                            env_state = self.scene.get_state()
                            env_state_json = sim_state_to_json(env_state)
                            sim_state = {"state": env_state_json}
                            self.recorder.save_data(
                                sim_state,
                                proprio_data=proprio_data,
                                action=pd_target,
                                action_mimic=action_mimic,
                                action_left_hand_mimic=action_left_hand,
                                action_right_hand_mimic=action_right_hand,
                            )
                            
                            # 触发停止：按下 B 键，立即切断 recording 信号
                            if button_b_pressed:
                                print("Stop recording episode and start saving.")
                                self.recorder.save_episode()
                                recording = False
                                saving = True # 标记进入后台保存等待期

                        if saving and self.recorder.is_available():
                            saving = False
                            print("Episode saved successfully, ready for next.")

                    cur_time = time.time()
                    if next_policy_deadline is None:
                        next_policy_deadline = cur_time + target_policy_period
                    else:
                        sleep_time = next_policy_deadline - cur_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                            next_policy_deadline += target_policy_period
                        else:
                            next_policy_deadline = cur_time + target_policy_period

                    cur_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = cur_time - last_policy_time
                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = float(np.mean(recent_intervals))
                            avg_execution_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
                            print(
                                f"Policy FPS (last {policy_fps_print_interval}): "
                                f"{avg_execution_fps:.2f} Hz "
                                f"(cycle {avg_interval * 1000:.2f}ms, "
                                f"compute {compute_elapsed * 1000:.2f}ms)"
                            )
                    last_policy_time = cur_time

        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()


    def cleanup(self):
        """安全关闭所有资源"""
        print("Cleaning up resources...")
        self._running = False

        if hasattr(self, 'recorder') and self.recorder is not None:
            try:
                self.recorder.close()
                print("Recorder saved.")
            except:
                pass

        stop_async_writer()
        
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except Exception:
            os._exit(0)


def main():

    try:
        if args.controller == "teleopit":
            tracker = partial(
                TeleopitController,
                vel_smoothing_alpha=0.35,
                anchor_vel_smoothing_alpha=0.25,
                qpos_smoothing_alpha=0.4,
            )
        else:
            print(f"Error: Controller {args.controller} does not exist")

        if args.record:
            from tools import camera_state as cam_state
            enable_jpeg = bool(args.camera_jpeg)
            jpeg_quality = int(args.camera_jpeg_quality)
            cam_state.set_writer_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality,
                                         skip_cvtcolor=args.skip_cvtcolor)


        if args.pico_host:
            from tools import camera_state as cam_state
            cam_state.enable_pico_streaming(
                host=args.pico_host,
                port=args.pico_port,
                fps=args.pico_fps,
                bitrate=args.pico_bitrate,
            )

        if args.replay:
            if args.input_dir is None:
                logging.warning("Data input_dir is empty!")
            else:
                replayer = Replayer(
                    args.input_dir,
                    args.output_dir,
                    num_envs=args.num_envs,
                    target_envs_per_episode=args.target_envs_per_episode,
                    device=args.device,
                    start=args.start,
                    end=args.end,
                )
                replayer.replay()
        else:
            sim_cfg = sim_utils.SimulationCfg(device=args.device)
            sim_cfg.dt = args.dt
            sim = SimulationContext(sim_cfg)
            scene_cfg = MotionsSceneCfgFactory.create(scene_type=args.scene, num_envs=1, env_spacing=2.0)
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            sim.set_camera_view(eye=[-2.46, -4.0, 1.8], target=[-2.46, -2.0, 0.8])

            controller = RealTimePolicyController(
                sim=sim,
                scene=scene,
                policy_path=args.policy,
                controller=tracker,
                device=args.device,
                record=args.record,
                record_dir=args.record_dir,
                robot_name=args.robot,
                dt=sim_cfg.dt,
                decimation=args.decimation,
                redis_host=args.redis_ip,
                redis_port=args.redis_port,
                redis_db=args.redis_db,
            )

            controller.run()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"\n[Main] Unexpected error: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
