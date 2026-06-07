# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue
from isaaclab.scene import InteractiveScene
# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tasks.teleop.tools.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")


def enable_pico_streaming(host: str, port: int = 12345, fps: int = 50, bitrate: int = 4_000_000) -> None:
    """Start streaming the front_camera RGB frames to a PICO headset.

    Idempotent: calling twice is a no-op after the first call.
    """
    global _pico_streamer
    if _pico_streamer is not None:
        return
    try:
        from tasks.teleop.tools.pico_streamer import PicoStreamer
        _pico_streamer = PicoStreamer(host=host, port=port, fps=fps, bitrate=bitrate)
        _pico_streamer.start()
        print(f"[camera_state] PICO streaming enabled -> {host}:{port} @ {fps}fps, {bitrate/1e6:.1f}Mbps")
    except Exception as e:
        print(f"[camera_state] failed to enable PICO streaming: {e}")
        _pico_streamer = None


def disable_pico_streaming() -> None:
    global _pico_streamer
    if _pico_streamer is not None:
        try:
            _pico_streamer.stop()
        except Exception as e:
            print(f"[camera_state] error stopping PICO streamer: {e}")
        _pico_streamer = None
        print("[camera_state] PICO streaming disabled.")


_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 2,
}

_async_queue = None
_async_thread = None
_async_started = False

# opt-in PICO live preview streamer (default OFF — replay/deploy callers that share
# this module must not accidentally start a TCP server). Enable via enable_pico_streaming().
_pico_streamer = None

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


def get_camera_image(
    scene: InteractiveScene, dt: float, replay_mode=False, num_envs=1
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """

    num_envs = 1 if not replay_mode else num_envs

    scene_id = id(scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        _camera_cache['last_scene_id'] = scene_id


    try:
        if hasattr(scene, 'sensors') and scene.sensors:
            for sensor in scene.sensors.values():
                try:
                    step_dt = sensor.cfg.update_period if replay_mode else dt
                    sensor.update(step_dt, force_recompute=False)
                except Exception:
                    pass
    except Exception:
        pass
    
    # get the camera images
    images = {}
    
    camera_keys = _camera_cache['camera_keys']
    available_cameras = _camera_cache['available_cameras']
    def extract_env_images(env_idx):
        img_dict = {}
        
        # 标准相机提取
        mapping = {
            "front_camera": "head",
            "left_wrist_camera": "left",
            "right_wrist_camera": "right"
        }
        
        for cam_key, label in mapping.items():
            if cam_key in camera_keys:
                img_dict[label] = scene[cam_key].data.output["rgb"][env_idx].contiguous().cpu().numpy()
        
        if not img_dict and available_cameras:
            for i, cam_name in enumerate(available_cameras):
                raw_img = scene[cam_name].data.output["rgb"][env_idx]
                label = ["head", "left", "right"][i]
                img_dict[label] = raw_img.cpu().numpy() if raw_img.device.type != 'cpu' else raw_img.numpy()
        
        return img_dict
    
    if not replay_mode:
        # 实时预览模式：只取环境 0，并尝试写入异步队列
        images = extract_env_images(0)
        if images:
            _ensure_async_started()
            try:
                if _async_queue.full():
                    _async_queue.get_nowait()
                _async_queue.put_nowait(images)
            except Exception:
                pass
            if _pico_streamer is not None and "head" in images:
                _pico_streamer.push(images["head"])
        return images
    else:
        # Replay 模式：并行提取所有环境的图像
        all_env_images = []
        for i in range(num_envs):
            all_env_images.append(extract_env_images(i))
        return all_env_images

    
    


def stop_async_writer():
    global _async_started, _async_queue, _async_thread
    disable_pico_streaming()
    if _async_started and _async_queue is not None:
        print("[camera_state] Sending shutdown signal to camera writer...")
        try:
            while not _async_queue.empty():
                _async_queue.get_nowait()
        except:
            pass

        _async_queue.put(None)

        if _async_thread is not None:
            _async_thread.join(timeout=2.0)

        multi_image_writer.close()
        _async_started = False
        print("[camera_state] Camera writer thread stopped.")

