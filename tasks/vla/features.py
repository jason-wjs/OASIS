"""
Feature utilities for VLA deployment.

FeatureBuilder : sim state → 69D motion feature  (online, incremental)
FeatureDecoder : 69D predicted features → controller-neutral MotionPlan

The 69D feature layout matches tasks/scripts/data.py exactly:
  [0:4]   φ(r_t)   = [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]
  [4:5]   Δψ_t     = yaw_{t+1} - yaw_t
  [5:8]   Δp_local = Rz(yaw_t)ᵀ @ (pos_{t+1} - pos_t)
  [8:9]   h_t      = root height
  [9:38]  q_t      = 29D body joint positions  (MuJoCo body-only order)
  [38:67] Δq_t     = joint increments
  [67:68] lh_t     = left  hand gripper scalar
  [68:69] rh_t     = right hand gripper scalar
"""

from __future__ import annotations

import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from typing import Optional

from tasks.params import VLA_STRIDE, mujoco_joint_names
from tasks.teleop.control import HandTrajectory, MotionPlan, MotionTrajectoryChunk

# Reuse geometry helpers from the training data pipeline.
# Kept as module-level functions so they can also be unit-tested standalone.

# ── Slice constants (mirror data.py) ─────────────────────────────────────────
STATE_DIM      = 69
PHI_SLICE      = slice(0,  4)
DYAW_SLICE     = slice(4,  5)
DP_LOCAL_SLICE = slice(5,  8)
HEIGHT_SLICE   = slice(8,  9)
QPOS_SLICE     = slice(9,  38)
DQPOS_SLICE    = slice(38, 67)
LH_SLICE       = slice(67, 68)
RH_SLICE       = slice(68, 69)

DT_POLICY = 0.02 * VLA_STRIDE  # stride-scaled policy interval (e.g. 5×20ms = 100ms)


# ── Geometry helpers (identical to tasks/scripts/data.py) ────────────────────

def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo [w, x, y, z] → (roll, pitch, yaw) via intrinsic ZYX Euler."""
    xyzw = quat_wxyz[[1, 2, 3, 0]]
    zyx = Rotation.from_quat(xyzw).as_euler('ZYX', degrees=False)
    yaw, pitch, roll = zyx[0], zyx[1], zyx[2]
    return np.array([roll, pitch, yaw], dtype=np.float64)


def phi_encode(roll: float, pitch: float) -> np.ndarray:
    """Continuous trigonometric encoding of roll and pitch (4D)."""
    return np.array([
        np.sin(roll),
        np.cos(roll) - 1.0,
        np.sin(pitch),
        np.cos(pitch) - 1.0,
    ], dtype=np.float32)


def phi_decode(phi: np.ndarray) -> tuple[float, float]:
    """Inverse of phi_encode: (4D) → (roll, pitch).

    φ = [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]
    cos(roll) = φ[1] + 1,  roll = atan2(φ[0], φ[1] + 1)
    """
    sin_roll  = float(phi[0])
    cos_roll  = float(phi[1]) + 1.0
    sin_pitch = float(phi[2])
    cos_pitch = float(phi[3]) + 1.0
    roll  = np.arctan2(sin_roll, cos_roll)
    pitch = np.arctan2(sin_pitch, cos_pitch)
    return roll, pitch


def wrap_angle(a: float) -> float:
    """Wrap angle difference to [-π, π]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def rz_transpose(yaw: float) -> np.ndarray:
    """3×3 matrix Rz(yaw)ᵀ — rotates world vector into yaw-aligned local frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c,  s,  0.0],
        [-s,  c,  0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rz_matrix(yaw: float) -> np.ndarray:
    """3×3 matrix Rz(yaw) — rotates local vector into world frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c, -s,  0.0],
        [ s,  c,  0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) → quaternion [w, x, y, z] (MuJoCo convention)."""
    r = Rotation.from_euler('ZYX', [yaw, pitch, roll])
    xyzw = r.as_quat()  # scipy → [x, y, z, w]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def encode_motion_features(
    root_positions: np.ndarray,
    root_quaternions_wxyz: np.ndarray,
    body_joint_positions: np.ndarray,
    left_hand_scalars: np.ndarray,
    right_hand_scalars: np.ndarray,
    stride: int = VLA_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a recorded trajectory into the canonical 69D VLA representation.

    Returns the feature matrix and the source-frame indices associated with
    each feature. The last stride frames have no future delta and are omitted.
    """
    positions = np.asarray(root_positions, dtype=np.float64)
    quaternions = np.asarray(root_quaternions_wxyz, dtype=np.float64)
    joints = np.asarray(body_joint_positions, dtype=np.float32)
    left = np.asarray(left_hand_scalars, dtype=np.float32)
    right = np.asarray(right_hand_scalars, dtype=np.float32)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    frame_count = positions.shape[0]
    expected = {
        "root_positions": (frame_count, 3),
        "root_quaternions_wxyz": (frame_count, 4),
        "body_joint_positions": (frame_count, 29),
        "left_hand_scalars": (frame_count,),
        "right_hand_scalars": (frame_count,),
    }
    actual = {
        "root_positions": positions.shape,
        "root_quaternions_wxyz": quaternions.shape,
        "body_joint_positions": joints.shape,
        "left_hand_scalars": left.shape,
        "right_hand_scalars": right.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")
    if frame_count <= stride:
        return np.empty((0, STATE_DIM), dtype=np.float32), np.empty(0, dtype=np.int64)

    sample_indices = np.arange(0, frame_count - stride, stride, dtype=np.int64)
    rpy = np.stack([quat_to_rpy(q) for q in quaternions])
    features = np.zeros((sample_indices.size, STATE_DIM), dtype=np.float32)
    for out_idx, frame_idx in enumerate(sample_indices):
        next_idx = frame_idx + stride
        roll, pitch, yaw = rpy[frame_idx]
        features[out_idx, PHI_SLICE] = phi_encode(roll, pitch)
        features[out_idx, DYAW_SLICE] = wrap_angle(rpy[next_idx, 2] - yaw)
        features[out_idx, DP_LOCAL_SLICE] = (
            rz_transpose(yaw) @ (positions[next_idx] - positions[frame_idx])
        )
        features[out_idx, HEIGHT_SLICE] = positions[frame_idx, 2]
        features[out_idx, QPOS_SLICE] = joints[frame_idx]
        features[out_idx, DQPOS_SLICE] = joints[next_idx] - joints[frame_idx]
        features[out_idx, LH_SLICE] = left[frame_idx]
        features[out_idx, RH_SLICE] = right[frame_idx]
    return features, sample_indices


# ── FeatureBuilder ───────────────────────────────────────────────────────────

class FeatureBuilder:
    """Build 69D motion features incrementally from sim states.

    Call ``update(...)`` every policy step (50 Hz).  With VLA_STRIDE > 1,
    a feature is produced only every ``stride`` calls.  Absolute values come
    from the stride-window start frame; delta values span the full window.

    Returns ``None`` on calls that don't produce a feature.
    """

    def __init__(self, stride: int = VLA_STRIDE) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._stride = stride
        self._call_count = 0

        # Start-of-window state (for absolute values + delta computation)
        self._window_pos:  Optional[np.ndarray] = None
        self._window_quat: Optional[np.ndarray] = None
        self._window_rpy:  Optional[np.ndarray] = None
        self._window_qpos: Optional[np.ndarray] = None
        self._window_lh: float = 0.0
        self._window_rh: float = 0.0

    def reset(self) -> None:
        self._call_count = 0
        self._window_pos = None
        self._window_quat = None
        self._window_rpy = None
        self._window_qpos = None
        self._window_lh = 0.0
        self._window_rh = 0.0

    def update(
        self,
        root_pos: np.ndarray,       # (3,)
        root_quat: np.ndarray,      # (4,) [w, x, y, z]
        body_qpos_29: np.ndarray,   # (29,) MuJoCo body-only order
        lh_scalar: float = 0.0,
        rh_scalar: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Ingest one frame.  Returns a 69D feature every ``stride`` calls."""
        # Validate quaternion
        qnorm = np.linalg.norm(root_quat)
        if not (0.99 < qnorm < 1.01):
            print(f"[FeatureBuilder] WARNING: root_quat norm={qnorm:.4f}, "
                  f"expected ~1.0 [w,x,y,z]. Normalizing.")
            root_quat = root_quat / qnorm

        self._call_count += 1

        # ── First call ever: store as window start, no feature yet ────────
        if self._window_pos is None:
            self._window_pos  = root_pos.copy()
            self._window_quat = root_quat.copy()
            self._window_rpy  = quat_to_rpy(root_quat)
            self._window_qpos = body_qpos_29.copy()
            self._window_lh   = lh_scalar
            self._window_rh   = rh_scalar
            self._call_count  = 0  # reset so stride counting starts fresh
            return None

        # ── Not yet at stride boundary: skip ──────────────────────────────
        if self._call_count < self._stride:
            return None

        # ── Stride boundary reached: compute feature ──────────────────────
        self._call_count = 0
        rpy = quat_to_rpy(root_quat)

        prev_roll, prev_pitch, prev_yaw = self._window_rpy
        curr_yaw = rpy[2]

        feature = np.zeros(STATE_DIM, dtype=np.float32)
        feature[PHI_SLICE]      = phi_encode(prev_roll, prev_pitch)
        feature[DYAW_SLICE]     = wrap_angle(curr_yaw - prev_yaw)
        feature[DP_LOCAL_SLICE] = (rz_transpose(prev_yaw) @ (root_pos - self._window_pos)).astype(np.float32)
        feature[HEIGHT_SLICE]   = self._window_pos[2]
        feature[QPOS_SLICE]     = self._window_qpos
        feature[DQPOS_SLICE]    = (body_qpos_29 - self._window_qpos).astype(np.float32)
        feature[LH_SLICE]       = self._window_lh
        feature[RH_SLICE]       = self._window_rh

        # Current frame becomes the start of the next window
        self._window_pos  = root_pos.copy()
        self._window_quat = root_quat.copy()
        self._window_rpy  = rpy.copy()
        self._window_qpos = body_qpos_29.copy()
        self._window_lh   = lh_scalar
        self._window_rh   = rh_scalar

        return feature


# ── HistoryBuffer ────────────────────────────────────────────────────────────

class HistoryBuffer:
    """Fixed-size ring buffer for normalized feature history.

    When fewer than ``maxlen`` frames are available, the buffer is left-padded
    with copies of the first frame (matching training-time behaviour).
    """

    def __init__(self, maxlen: int = 2) -> None:
        self.maxlen = maxlen
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)

    def reset(self) -> None:
        self._buf.clear()

    def append(self, feature: np.ndarray) -> None:
        self._buf.append(feature.copy())

    def __len__(self) -> int:
        return len(self._buf)

    def ready(self) -> bool:
        """At least one frame available (can pad the rest)."""
        return len(self._buf) > 0

    def get_padded(self) -> np.ndarray:
        """Return (maxlen, 69) array, left-padded with first frame if needed."""
        n = len(self._buf)
        if n == 0:
            raise RuntimeError("HistoryBuffer is empty — call append() first")

        frames = list(self._buf)
        if n < self.maxlen:
            pad_count = self.maxlen - n
            frames = [frames[0]] * pad_count + frames

        return np.stack(frames, axis=0)  # (maxlen, 69)


# ── FeatureDecoder ───────────────────────────────────────────────────────────

class FeatureDecoder:
    """Decode VLA-predicted 69D features into a controller-neutral motion plan.

    The predicted features are in *normalized* space; the decoder first
    de-normalizes using the training norm_stats, then reconstructs absolute
    root state by integrating Δψ and Δp from the known current state.
    """

    def __init__(
        self,
        norm_stats: dict,
        hand_open_left: np.ndarray,
        hand_close_left: np.ndarray,
        hand_open_right: np.ndarray,
        hand_close_right: np.ndarray,
        dt_policy: float = DT_POLICY,
    ) -> None:
        self.state_mean = norm_stats['state_mean'].astype(np.float32)
        self.state_std = np.maximum(norm_stats['state_std'].astype(np.float32), 1e-8)
        if self.state_mean.shape != (STATE_DIM,) or self.state_std.shape != (STATE_DIM,):
            raise ValueError("normalization statistics must both have shape (69,)")
        if dt_policy <= 0:
            raise ValueError("dt_policy must be positive")
        self.hand_open_left = np.asarray(hand_open_left, dtype=np.float32)
        self.hand_close_left = np.asarray(hand_close_left, dtype=np.float32)
        self.hand_open_right = np.asarray(hand_open_right, dtype=np.float32)
        self.hand_close_right = np.asarray(hand_close_right, dtype=np.float32)
        if any(pose.shape != (7,) for pose in (
            self.hand_open_left, self.hand_close_left,
            self.hand_open_right, self.hand_close_right,
        )):
            raise ValueError("all hand poses must have shape (7,)")
        self.dt_policy = float(dt_policy)

    def _denormalize(self, features: np.ndarray) -> np.ndarray:
        result = features * self.state_std + self.state_mean
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print("[FeatureDecoder] WARNING: NaN/Inf after denormalization, clamping")
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        return result

    def _hand_scalar_to_7d(self, scalar: float, side: str) -> np.ndarray:
        """Linear interpolation between open and close hand poses.

        Convention (from data_json_load.py preprocessing):
          0.0 = open  (all joints at zero)
          1.0 = close (joints engaged)
        If training convention is inverted, change to: s = 1.0 - s
        """
        s = float(np.clip(scalar, 0.0, 1.0))
        if side == 'left':
            return self.hand_open_left * (1.0 - s) + self.hand_close_left * s
        else:
            return self.hand_open_right * (1.0 - s) + self.hand_close_right * s

    def decode_sequence(
        self,
        predicted_features: np.ndarray,   # (future_len, 69) normalized
        current_pos: np.ndarray,           # (3,)
        current_yaw: float,
        start_time_s: float = 0.0,
        source: str = "vla",
    ) -> MotionPlan:
        """Decode normalized future features into absolute motion references."""
        predicted = np.asarray(predicted_features, dtype=np.float32)
        if predicted.ndim != 2 or predicted.shape[1] != STATE_DIM or predicted.shape[0] == 0:
            raise ValueError(
                f"predicted_features must have shape (F, {STATE_DIM}) with F > 0"
            )
        pos = np.asarray(current_pos, dtype=np.float64)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("current_pos must be a finite array with shape (3,)")
        features = self._denormalize(predicted)
        frame_count = features.shape[0]
        joint_positions = np.empty((frame_count, 29), dtype=np.float32)
        joint_velocities = np.empty((frame_count, 29), dtype=np.float32)
        root_positions = np.empty((frame_count, 3), dtype=np.float32)
        root_quaternions = np.empty((frame_count, 4), dtype=np.float32)
        left_hands = np.empty((frame_count, 7), dtype=np.float32)
        right_hands = np.empty((frame_count, 7), dtype=np.float32)

        pos = pos.copy()
        yaw = float(current_yaw)
        max_delta_yaw = 0.5 * (self.dt_policy / 0.02)
        for index, feat in enumerate(features):
            joint_positions[index] = feat[QPOS_SLICE]
            joint_velocities[index] = feat[DQPOS_SLICE] / self.dt_policy
            height = float(feat[HEIGHT_SLICE][0])
            roll, pitch = phi_decode(feat[PHI_SLICE])
            delta_yaw = float(
                np.clip(feat[DYAW_SLICE][0], -max_delta_yaw, max_delta_yaw)
            )
            delta_p_local = feat[DP_LOCAL_SLICE].astype(np.float64)
            delta_p_world = rz_matrix(yaw) @ delta_p_local
            pos += delta_p_world
            yaw = wrap_angle(yaw + delta_yaw)

            root_pos = pos.copy()
            root_pos[2] = height
            root_positions[index] = root_pos
            root_quaternions[index] = rpy_to_quat_wxyz(roll, pitch, yaw)
            left_hands[index] = self._hand_scalar_to_7d(feat[LH_SLICE][0], 'left')
            right_hands[index] = self._hand_scalar_to_7d(feat[RH_SLICE][0], 'right')

        timestamps = start_time_s + self.dt_policy * np.arange(
            1, frame_count + 1, dtype=np.float64
        )
        motion = MotionTrajectoryChunk(
            timestamps_s=timestamps,
            joint_names=mujoco_joint_names,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            root_positions=root_positions,
            root_quaternions_wxyz=root_quaternions,
        )
        hands = HandTrajectory(
            timestamps_s=timestamps,
            left_joint_positions=left_hands,
            right_joint_positions=right_hands,
        )
        return MotionPlan(
            motion=motion,
            hands=hands,
            source=source,
            metadata={"feature_codec": "oasis-vla-69d-v1"},
        )
