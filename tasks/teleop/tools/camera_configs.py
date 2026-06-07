# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
public camera configuration
include the basic configuration for different types of cameras, support scene-specific parameter customization
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
import os
import torch
import numpy as np
@configclass
class CameraBaseCfg:
    """camera base configuration class
    
    provide the default configuration for different types of cameras, support scene-specific parameter customization
    """
    
    @classmethod
    def get_camera_config(
        cls,
        prim_path: str = "{ENV_REGEX_NS}/Robot/d435_link/front_camera",
        update_period: float = 0.02,
        height: int = 480,
        width: int =  640,
        focal_length: float = 7.6,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.0,
        vertical_aperture: float = 15.0,
        clipping_range: tuple = (0.1, 1.0e5),
        pos_offset: tuple = (0, 0.0, 0),
        rot_offset: tuple = (0.5, -0.5, 0.5, -0.5),
        data_types: list = None
    ) -> CameraCfg:
        """get the front camera configuration
        
        Args:
            prim_path: the path of the camera in the scene
            update_period: update period (seconds)
            height: image height (pixels)
            width: image width (pixels)
            focal_length: focal length
            focus_distance: focus distance
            horizontal_aperture: horizontal aperture
            clipping_range: clipping range (near clipping plane, far clipping plane)
            pos_offset: position offset (x, y, z)
            rot_offset: rotation offset quaternion
            data_types: data type list
            
        Returns:
            CameraCfg: camera configuration
        """
        if data_types is None:
            data_types = ["rgb"]

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture,
                clipping_range=clipping_range
            ),
            offset=CameraCfg.OffsetCfg(
                pos=pos_offset,
                rot=rot_offset,
                convention="opengl" # TODO:改成opengl的
            )
        )
    



@configclass
class CameraPresets:
    """camera preset configuration collection
    
    include the common camera configuration preset for different scenes
    """
    
    @classmethod
    def g1_front_camera(cls) -> CameraCfg:

        """front camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="{ENV_REGEX_NS}/Robot/d435_link/front_camera",
            height=480,
            width=640,
            update_period=0.02,
            data_types=["rgb"],
            focal_length=1.0,
            focus_distance=400.0,
            horizontal_aperture=1.05508,
            vertical_aperture=0.79239,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(0.0, 0.0, 0.0),
            rot_offset=(0.5, 0.5, -0.5, -0.5)
        )
    @classmethod
    def left_dex3_wrist_camera(cls) -> CameraCfg:
        """left wrist camera configuration"""
        # Realsense D405
        return CameraBaseCfg.get_camera_config(
            prim_path="{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_camera",
            height=480,
            width=640,
            update_period=0.02,
            data_types=["rgb"],
            focal_length=1,
            focus_distance=0.3,
            horizontal_aperture=1.62807,
            vertical_aperture=1.22427,
            clipping_range=(0.05, 100),
            pos_offset=(-0.02341, -0.01756, 0.08963),
            rot_offset=(0.44837,0.3002,-0.41726, -0.73126),
        )
    @classmethod
    def right_dex3_wrist_camera(cls) -> CameraCfg:
        """right wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_camera",
            height=480,
            width=640,
            update_period=0.02,
            data_types=["rgb"],
            focal_length=1,
            focus_distance=0.3,
            horizontal_aperture=1.62807,
            vertical_aperture=1.22427,
            clipping_range=(0.05, 100.0),
            pos_offset=(-0.02341, 0.01756 ,0.08963),
            rot_offset=(0.73126,0.41726,-0.3002, -0.44837),
        ) 
    
