from typing import Optional
import torch
from tasks.teleop.tools.shared_memory_utils import MultiImageReader
from tasks.teleop.tools.episode_writer import EpisodeWriter
import json
from typing import List, Optional
import numpy as np
import time
import cv2
class StateRecorder:
    def __init__(self, record_dir, frequency=50, skip_json=False, text: Optional[str] = None):

        try:
            self.multi_image_reader = MultiImageReader()
            print("[Recorder] MultiImageReader created")
        except Exception as e:
            print("[Recorder] MultiImageReader creation failed: {e}")
            print("[Recorder] Image data saving will be disabled")
            self.multi_image_reader = None

        self.writer = EpisodeWriter(task_dir=record_dir, frequency=frequency, skip_json=skip_json, text=text)

    def cleanup(self):
        """Clean up DDS resources"""
        if self.multi_image_reader:
            self.multi_image_reader.close()
        if self.writer:
            self.writer.close()
        self.is_running = False
        print("[Recorder] Resource cleanup completed")

    def get_images(self, names):
        """Get images using read_single_image for each camera (no merge/split operations)"""
        images = {}
        
        for name in names:
            image = self.multi_image_reader.read_single_image(name)
            if image is not None:
                images[name] = image
            else:
                print(f"Warning: {name} image not available in shared memory")

        # Check if we have the expected number of images
        if len(images) != len(names):
            print(f"Warning: Expected {len(names)} images, got {len(images)}")
            # For backward compatibility, return None if not all images are available
            return None

        return images

    def save_episode(self):
        self.writer.save_episode()

    def is_available(self):
        return self.writer.is_available

    def create_episode(self, scene, save_dir=None):
        return self.writer.create_episode(scene=scene, save_dir=save_dir)

    def close(self):
        self.writer.close()

    def save_data(self, sim_state=None, proprio_data=None, action=None, action_mimic=None, action_left_hand_mimic=None, action_right_hand_mimic=None, replay_mode=False, camera_images=None, camera_order=['head', 'left', 'right']):
        def ensure_list(data):
            """Ensure data is list type, if not, convert to list"""
            if isinstance(data, list):
                return data
            elif hasattr(data, 'tolist'):  # numpy array or torch tensor
                return data.tolist()
            elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):  # other iterable types
                return list(data)
            else:  # single valuerecord_dir
                return [data]

        states = None
        actions = None
        mimic = None
        images = None
        colors = {}

        if not replay_mode:
            root_pos, root_quat, lin_vel, ang_vel, dof_pos_full, dof_vel_full, torso_quat = proprio_data
            dof_pos = np.concatenate([dof_pos_full[:22], dof_pos_full[29:36]])
            dof_vel = np.concatenate([dof_vel_full[:22], dof_vel_full[29:36]])
            left_hand_pos = dof_pos_full[22:29]
            left_hand_vel = dof_vel_full[22:29]
            right_hand_pos = dof_pos_full[36:]
            right_hand_vel = dof_vel_full[36:]

            t = time.time()

            # camera_list = ['head']
            # images = self.get_images(camera_list)
            # if images is None:
            #     return False

            # for i in range(len(camera_list)):
            #     colors[f"color_{i}"] = images[camera_list[i]]

            left_hand_action = action[22:29].tolist()
            right_hand_action = action[36:].tolist()
            body_action = np.concatenate([action[:22], action[29:36]])

            states = {
                "root":{
                    "pos": ensure_list(root_pos),
                    "quat": ensure_list(root_quat)
                },
                "body": {
                    "qpos": ensure_list(dof_pos),
                    "qvel": ensure_list(dof_vel),
                    "lin_vel": ensure_list(lin_vel),
                    "ang_vel": ensure_list(ang_vel),
                },
                "left_hand": {
                    "qpos":   ensure_list(left_hand_pos),
                    "qvel":   ensure_list(left_hand_vel),
                },
                "right_hand": {
                    "qpos":   ensure_list(right_hand_pos),
                    "qvel":   ensure_list(right_hand_vel),
                },

            }
            actions = {
                "body": {
                    "qpos": ensure_list(body_action),
                },
                "left_hand": {
                    "qpos": ensure_list(left_hand_action),
                },
                "right_hand": {
                    "qpos": ensure_list(right_hand_action),
                },

            }

            if action_mimic is not None:
                mimic = {
                    "body": {
                        "joint_pos": ensure_list(action_mimic.get("joint_pos", [])),
                        "joint_vel": ensure_list(action_mimic.get("joint_vel", [])),
                        "root_pos":  ensure_list(action_mimic.get("root_pos",  [])),
                        "root_quat": ensure_list(action_mimic.get("root_quat", [])),
                    },
                    "left_hand": {
                        "qpos": ensure_list(action_left_hand_mimic) if action_left_hand_mimic is not None else [],
                    },
                    "right_hand": {
                        "qpos": ensure_list(action_right_hand_mimic) if action_right_hand_mimic is not None else [],
                    },
                }
        else:
            if camera_images is not None:
                # 按照固定顺序处理，保证数据集 color_0/1/2 不会跳变
                
                
                for i, cam_name in enumerate(camera_order):
                    if cam_name in camera_images:
                        img_np = camera_images[cam_name]
                        
                        # --- 解决失真：处理 float32 [0, 1] 数据 ---
                        if img_np.dtype != np.uint8:
                            # 强制映射到 0-255 并转为 uint8
                            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                        
                        # 确保内存连续性，防止 cv2 处理花屏
                        img_np = np.ascontiguousarray(img_np)

                        # --- 颜色纠偏：RGB -> BGR ---
                        # Isaac Lab 原生是 RGB，cv2.imwrite 需要 BGR
                        if img_np.ndim == 3 and img_np.shape[2] == 3:
                            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img_np
                        
                        colors[f"color_{i}"] = img_bgr



        self.writer.add_item(colors=colors, states=states, actions=actions, action_mimic=mimic, sim_state=sim_state)
        return True

    def clear(self):
        self.writer.clear_queue()
        
    def sim_state_to_json(self ,data):
        data_serializable = self.tensors_to_list(data)
        json_str = json.dumps(data_serializable)
        return json_str
    def tensors_to_list(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensors_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensors_to_list(i) for i in obj]
        return obj