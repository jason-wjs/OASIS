import pathlib

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

ASSET_ROOT = PROJECT_ROOT / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
}

ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
}

REDIS_ACTION_KEY = "action_payload_unitree_g1"

# VLA motion features are sampled at the 50 Hz low-level policy rate. Keep
# this explicit because it is part of both the dataset and deployment codec.
VLA_STRIDE = 1

# --- Default mimic observation / hand poses (merged from data_utils/params.py) ---

DEFAULT_OBS_G1 = {
                    "joint_pos":[-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                            0.0, 0.0, 0.0, # torso (1)
                            #0.2, 0.2, 0, 0.9, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, -0.4, 0.0, 0.0, 0.0, # left arm (7)
                            0.0, 0.0, 0.0, -0.4, 0.0, 0.0, 0.0, # right arm (7)
                        ],
                    # 29 dof
                    "joint_vel":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left leg (6)
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right leg (6)
                            0.0, 0.0, 0.0, # torso (1)
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # left arm (7)
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # right arm (7)
                        ],
                    "root_pos": [0.0, 0.0, 0.78],
                    "root_quat": [1.0, 0.0, 0.0, 0.0],
                    "root_lin_vel_w": [0.0, 0.0, 0.0],
                    "root_ang_vel_w": [0.0, 0.0, 0.0],
}

DEFAULT_MIMIC_OBS = {
    "unitree_g1": DEFAULT_OBS_G1,
}


DEFAULT_HAND_POSE = {
    "unitree_g1":
    {
        "left": {
            "open": np.array([0.0, 0.0, 0.0, 0, 0, 0, 0]),
            "close": np.array([
                    # left (thumb, middle, index)
                    0.0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74,
                ]),
        },
        "right": {
            "open": np.array([0.0, 0, 0, 0, 0, 0, 0]),
            "close": np.array([
                    # right (thumb, middle, index)
                    0.0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74,
                ]),
        },
    },
}

isaaclab_g1_hand_joint_names = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint',
    'left_hand_index_0_joint',
    'left_hand_middle_0_joint',
    'left_hand_thumb_0_joint',
    'right_hand_index_0_joint',
    'right_hand_middle_0_joint',
    'right_hand_thumb_0_joint',
    'left_hand_index_1_joint',
    'left_hand_middle_1_joint',
    'left_hand_thumb_1_joint',
    'right_hand_index_1_joint',
    'right_hand_middle_1_joint',
    'right_hand_thumb_1_joint',
    'left_hand_thumb_2_joint',
    'right_hand_thumb_2_joint'
]

mujoco_g1_hand_joint_names = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'left_hand_thumb_0_joint',
    'left_hand_thumb_1_joint',
    'left_hand_thumb_2_joint',
    'left_hand_middle_0_joint',
    'left_hand_middle_1_joint',
    'left_hand_index_0_joint',
    'left_hand_index_1_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
    'right_hand_thumb_0_joint',
    'right_hand_thumb_1_joint',
    'right_hand_thumb_2_joint',
    'right_hand_middle_0_joint',
    'right_hand_middle_1_joint',
    'right_hand_index_0_joint',
    'right_hand_index_1_joint'
]


isaaclab_joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

mujoco_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


