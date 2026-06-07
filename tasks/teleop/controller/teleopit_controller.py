from collections import deque

import mujoco
import numpy as np
import torch

from tasks import ASSET_ROOT
from tasks.teleop.data_utils.rot_utils import (
    get_yaw_quat_only, quat_apply_numpy, quat_axis_angle_vel, quat_inv_numpy,
    quat_mul_numpy, quat_slerp_numpy, quat_to_rot6d,
)
from tasks.teleop.tools import load_onnx_policy

from .base_controller import BaseController

class ExponentialVecSmoother:
    """EMA smoother for vectors, alpha in (0, 1]; alpha=1.0 is a pass-through.

    Mirrors teleopit/sim/realtime_utils.py ExponentialVecSmoother.
    """

    def __init__(self, alpha: float):
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0 or a > 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = a
        self._state = None

    def reset(self):
        self._state = None

    def apply(self, value):
        cur = np.asarray(value, dtype=np.float32).reshape(-1)
        if self._state is None or self._state.shape != cur.shape or self._alpha >= 1.0 - 1e-6:
            self._state = cur.copy()
            return self._state.copy()
        self._state = ((1.0 - self._alpha) * self._state + self._alpha * cur).astype(np.float32)
        return self._state.copy()
    
class _QposLowPassFilter:
    """Low-pass filter on motion qpos: xyz / joints linear EMA, quat slerp.

    Mirrors teleopit/controllers/qpos_interpolator.py QposLowPassFilter.
    Runs BEFORE finite-diff so computed velocities are inherently smoother.
    """

    def __init__(self, alpha: float):
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0 or a > 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = a
        self._pos = None
        self._quat = None
        self._joints = None

    def reset(self):
        self._pos = None
        self._quat = None
        self._joints = None

    def apply(self, pos, quat, joints):
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        quat = np.asarray(quat, dtype=np.float64).reshape(-1)
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        if self._pos is None or self._alpha >= 1.0 - 1e-6:
            self._pos = pos.copy()
            self._quat = quat.copy()
            self._joints = joints.copy()
            return (pos.astype(np.float32), quat.astype(np.float32),
                    joints.astype(np.float32))
        a = self._alpha
        self._pos = (1.0 - a) * self._pos + a * pos
        self._quat = quat_slerp_numpy(self._quat, quat, a)
        self._joints = (1.0 - a) * self._joints + a * joints
        return (self._pos.astype(np.float32), self._quat.astype(np.float32),
                self._joints.astype(np.float32))


_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.4386, 0.4386,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
], dtype=np.float32)


class TeleopitController(BaseController):
    """Dual-input ONNX velcmd_history policy, 166D obs per frame."""

    def __init__(self, policy_path: str, device: str,
                 policy_hz: float = 50.0, history_len: int = 10,
                 vel_smoothing_alpha: float = 0.35,
                 anchor_vel_smoothing_alpha: float = 0.25,
                 qpos_smoothing_alpha: float = 0.4,
                 ):

        self.num_actions = 29
        self.default_dof_pos = DEFAULT_ANGLES.copy()
        self.action_scale = ACTION_SCALE.copy()
        self.device = device
        self._dt = 1.0 / float(policy_hz)
        self._max_ref_dt = 5.0 * self._dt
        self.total_obs_size = 166

        self.policy = load_onnx_policy(policy_path, device)
        raw_h = self.policy.input_shapes[1][1]
        self.history_len = raw_h if isinstance(raw_h, int) and raw_h > 0 else history_len

        # mujoco kinematics (motion-side torso FK). Only needed in torso mode.
        self._mj_model = None
        self._mj_data = None
        self._torso_body_id = -1
        
        mj_xml = f"{ASSET_ROOT}/unitree_g1/g1_mocap_29dof.xml"
        self._mj_model = mujoco.MjModel.from_xml_path(mj_xml)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._torso_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        if self._torso_body_id < 0:
            raise RuntimeError("torso_link body not found in g1_mocap_29dof.xml")

        self._qpos_filter = _QposLowPassFilter(qpos_smoothing_alpha)
        self._joint_vel_smoother = ExponentialVecSmoother(vel_smoothing_alpha)
        self._lin_vel_smoother = ExponentialVecSmoother(anchor_vel_smoothing_alpha)
        self._ang_vel_smoother = ExponentialVecSmoother(anchor_vel_smoothing_alpha)

        self._obs_history = deque(maxlen=self.history_len)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._init_ref_state()

    def _init_ref_state(self):
        """Zero out ref-tracking state (finite-diff prevs + stale-frame holds)."""
        self._q_offset = None  # yaw-only alignment quat, locked on first ref frame

        self._prev_motion_joint_pos = None
        self._prev_motion_torso_pos = None
        self._prev_motion_torso_quat = None
        self._prev_motion_payload_t = None

        self._last_payload_t = None
        self._last_raw_sig = None
        self._last_motion_joint_pos = None
        self._last_motion_root_q = None
        self._last_motion_torso_q_w = None
        self._last_motion_joint_vel = None
        self._last_motion_lin_w = None
        self._last_motion_ang_w = None

    def reset(self):
        self._obs_history.clear()
        self._last_action[:] = 0.0
        self._qpos_filter.reset()
        self._joint_vel_smoother.reset()
        self._lin_vel_smoother.reset()
        self._ang_vel_smoother.reset()
        self._init_ref_state()

    def _motion_torso_fk(self, root_pos_w, root_q_w, joint_pos_29):
        """Return (torso_pos_w, torso_q_w) for the motion pose via mujoco FK.

        root_q_w is wxyz. joint_pos_29 must be in MuJoCo joint order; the first
        12 entries are both legs (hip_pitch, hip_roll, hip_yaw, knee,
        ankle_pitch, ankle_roll) and entries [12:15] are the waist
        (yaw, roll, pitch) that drive the pelvis→torso_link chain.
        """
        self._mj_data.qpos[:] = 0.0
        self._mj_data.qpos[0:3] = np.asarray(root_pos_w, dtype=np.float64)
        q = np.asarray(root_q_w, dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        self._mj_data.qpos[3:7] = q
        n = min(len(joint_pos_29), self._mj_model.nq - 7)
        self._mj_data.qpos[7:7 + n] = np.asarray(joint_pos_29, dtype=np.float64)[:n]
        mujoco.mj_kinematics(self._mj_model, self._mj_data)
        pos = np.asarray(self._mj_data.xpos[self._torso_body_id], dtype=np.float32).copy()
        quat = np.asarray(self._mj_data.xquat[self._torso_body_id], dtype=np.float32).copy()
        return pos, quat

    def _build_obs(self, proprio_data, ref_data, start_receive_ref_data):
        (_, root_quat, _, ang_vel_b, dof_pos_full, dof_vel_full, torso_quat) = proprio_data

        joint_pos = np.concatenate([dof_pos_full[:22], dof_pos_full[29:36]]).astype(np.float32)
        joint_vel = np.concatenate([dof_vel_full[:22], dof_vel_full[29:36]]).astype(np.float32)

        robot_root_q = np.asarray(root_quat, dtype=np.float32)
        robot_torso_q = np.asarray(torso_quat, dtype=np.float32)
                         

        raw_motion_joint_pos = np.asarray(ref_data["joint_pos"], dtype=np.float32)
        raw_motion_root_pos = np.asarray(ref_data["root_pos"], dtype=np.float32)
        raw_motion_root_q = np.asarray(ref_data["root_quat"], dtype=np.float32)

        cur_t = ref_data.get("t", None)
        if cur_t is not None and self._last_payload_t is not None:
            is_stale = (cur_t == self._last_payload_t)
        else:
            raw_sig = (raw_motion_joint_pos.tobytes(),
                       raw_motion_root_pos.tobytes(),
                       raw_motion_root_q.tobytes())
            is_stale = (self._last_raw_sig is not None
                        and raw_sig == self._last_raw_sig)
            self._last_raw_sig = raw_sig
        if cur_t is not None:
            self._last_payload_t = cur_t

        if is_stale and self._last_motion_joint_pos is not None:
            motion_joint_pos = self._last_motion_joint_pos.copy()
            motion_root_q = self._last_motion_root_q.copy()
            motion_torso_q_w = self._last_motion_torso_q_w.copy()
            motion_joint_vel = self._last_motion_joint_vel.copy()
            motion_lin_w = self._last_motion_lin_w.copy()
            motion_ang_w = self._last_motion_ang_w.copy()
        else:
            if start_receive_ref_data and self._q_offset is None:
                self._prev_motion_joint_pos = None
                self._prev_motion_torso_pos = None
                self._prev_motion_torso_quat = None
                self._prev_motion_payload_t = None
                self._qpos_filter.reset()
                self._joint_vel_smoother.reset()
                self._lin_vel_smoother.reset()
                self._ang_vel_smoother.reset()

            # 先平滑motion的pos
            motion_root_pos, motion_root_q, motion_joint_pos = self._qpos_filter.apply(
                raw_motion_root_pos, raw_motion_root_q, raw_motion_joint_pos
            )

            
            motion_torso_pos_w, motion_torso_q_w = self._motion_torso_fk(
                motion_root_pos, motion_root_q, motion_joint_pos
            )

            if cur_t is not None and self._prev_motion_payload_t is not None:
                dt_real = float(cur_t - self._prev_motion_payload_t)
            else:
                dt_real = self._dt
            dt_is_sane = np.isfinite(dt_real) and (1e-4 < dt_real <= self._max_ref_dt)

            if self._prev_motion_joint_pos is None:
                raw_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
                raw_lin_vel_w = np.zeros(3, dtype=np.float32)
                raw_ang_vel_w = np.zeros(3, dtype=np.float32)
            elif not dt_is_sane:
                raw_joint_vel = (self._last_motion_joint_vel.copy()
                                 if self._last_motion_joint_vel is not None
                                 else np.zeros(self.num_actions, dtype=np.float32))
                raw_lin_vel_w = (self._last_motion_lin_w.copy()
                                 if self._last_motion_lin_w is not None
                                 else np.zeros(3, dtype=np.float32))
                raw_ang_vel_w = (self._last_motion_ang_w.copy()
                                 if self._last_motion_ang_w is not None
                                 else np.zeros(3, dtype=np.float32))
            else:
                inv_dt = np.float32(1.0 / dt_real)
                raw_joint_vel = ((motion_joint_pos - self._prev_motion_joint_pos)
                                 * inv_dt).astype(np.float32)
                raw_lin_vel_w = ((motion_torso_pos_w - self._prev_motion_torso_pos)
                                 * inv_dt).astype(np.float32)
                raw_ang_vel_w = quat_axis_angle_vel(
                    motion_torso_q_w, self._prev_motion_torso_quat, dt_real
                )
            motion_joint_vel = self._joint_vel_smoother.apply(raw_joint_vel)
            motion_lin_w = self._lin_vel_smoother.apply(raw_lin_vel_w)
            motion_ang_w = self._ang_vel_smoother.apply(raw_ang_vel_w)

            self._prev_motion_joint_pos = motion_joint_pos.copy()
            self._prev_motion_torso_pos = motion_torso_pos_w.copy()
            self._prev_motion_torso_quat = motion_torso_q_w.copy()
            self._prev_motion_payload_t = cur_t
            self._last_motion_joint_pos = motion_joint_pos.copy()
            self._last_motion_root_q = motion_root_q.copy()
            self._last_motion_torso_q_w = motion_torso_q_w.copy()
            self._last_motion_joint_vel = motion_joint_vel.copy()
            self._last_motion_lin_w = motion_lin_w.copy()
            self._last_motion_ang_w = motion_ang_w.copy()


        if start_receive_ref_data and self._q_offset is None:
            delta_q = quat_mul_numpy(robot_root_q, quat_inv_numpy(motion_root_q))
            self._q_offset = get_yaw_quat_only(delta_q).astype(np.float32)

        if self._q_offset is not None:
            motion_torso_q_w = quat_mul_numpy(self._q_offset, motion_torso_q_w).astype(np.float32)
            motion_lin_w = quat_apply_numpy(self._q_offset, motion_lin_w).astype(np.float32)
            motion_ang_w = quat_apply_numpy(self._q_offset, motion_ang_w).astype(np.float32)


        command = np.concatenate([motion_joint_pos, motion_joint_vel])  # 58

        rel_q = quat_mul_numpy(quat_inv_numpy(robot_torso_q), motion_torso_q_w)
        motion_anchor_ori_b = quat_to_rot6d(rel_q)  # 6

        base_ang_vel = np.asarray(ang_vel_b, dtype=np.float32)
        joint_pos_rel = joint_pos - self.default_dof_pos
        projected_gravity = quat_apply_numpy(
            quat_inv_numpy(robot_root_q), _GRAVITY_W).astype(np.float32)

        robot_torso_inv = quat_inv_numpy(robot_torso_q)
        ref_base_lin_vel_b = quat_apply_numpy(robot_torso_inv, motion_lin_w).astype(np.float32)
        ref_base_ang_vel_b = quat_apply_numpy(robot_torso_inv, motion_ang_w).astype(np.float32)
        ref_projected_gravity_b = quat_apply_numpy(
            quat_inv_numpy(motion_torso_q_w), _GRAVITY_W).astype(np.float32)

        obs = np.concatenate([
            command,                   # 58
            motion_anchor_ori_b,       # 6
            base_ang_vel,              # 3
            joint_pos_rel,             # 29
            joint_vel,                 # 29
            self._last_action,         # 29
            projected_gravity,         # 3
            ref_base_lin_vel_b,        # 3
            ref_base_ang_vel_b,        # 3
            ref_projected_gravity_b,   # 3
        ]).astype(np.float32)
        if obs.shape[0] != self.total_obs_size:
            raise ValueError(f"Expected {self.total_obs_size}D, got {obs.shape[0]}")
        if not np.all(np.isfinite(obs)):
            obs = np.where(np.isfinite(obs), obs, np.float32(0.0))
        return obs

    def step(self, proprio_data, ref_data, start_receive_ref_data=False):
        obs = self._build_obs(proprio_data, ref_data, start_receive_ref_data)

        if len(self._obs_history) == 0:
            for _ in range(self.history_len):
                self._obs_history.append(obs.copy())
        else:
            self._obs_history.append(obs.copy())

        obs_tensor = torch.from_numpy(obs[None].astype(np.float32))                 # (1,166)
        hist_tensor = torch.from_numpy(
            np.stack(self._obs_history, axis=0)[None].astype(np.float32))           # (1,H,166)

        with torch.no_grad():
            raw_action = self.policy(obs_tensor, hist_tensor).cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(raw_action)):
            raw_action = np.zeros_like(raw_action)
        self._last_action = raw_action  # cache pre-clip for next-step obs

        clipped = np.clip(raw_action, -10.0, 10.0)
        return clipped * self.action_scale + self.default_dof_pos
