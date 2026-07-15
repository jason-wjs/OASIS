"""Controller-neutral motion plan data transfer objects.

The VLA layer predicts trajectory chunks; a selected low-level controller
adapts those chunks to its own observation and reference conventions. These
types intentionally contain no Teleopit, HEFT, SONIC, Isaac Lab, or Redis
details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class MotionTrajectoryChunk:
    """A time-indexed whole-body reference in one declared joint order."""

    timestamps_s: np.ndarray
    joint_names: Sequence[str]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    root_positions: np.ndarray
    root_quaternions_wxyz: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("timestamps_s must be a non-empty 1D array")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps_s must be finite and strictly increasing")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be non-empty and unique")

        frames, joints = timestamps.size, len(names)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(
            self,
            "joint_positions",
            _array(self.joint_positions, (frames, joints), "joint_positions"),
        )
        object.__setattr__(
            self,
            "joint_velocities",
            _array(self.joint_velocities, (frames, joints), "joint_velocities"),
        )
        object.__setattr__(
            self,
            "root_positions",
            _array(self.root_positions, (frames, 3), "root_positions"),
        )
        quaternions = _array(
            self.root_quaternions_wxyz, (frames, 4), "root_quaternions_wxyz"
        )
        norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("root_quaternions_wxyz must contain unit quaternions")
        object.__setattr__(self, "root_quaternions_wxyz", quaternions)

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_s.size)


@dataclass(frozen=True)
class HandTrajectory:
    """Time-indexed left and right hand joint targets."""

    timestamps_s: np.ndarray
    left_joint_positions: np.ndarray
    right_joint_positions: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("timestamps_s must be a non-empty 1D array")
        frames = timestamps.size
        left = np.asarray(self.left_joint_positions, dtype=np.float32)
        right = np.asarray(self.right_joint_positions, dtype=np.float32)
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[0] != frames
            or right.shape[0] != frames
        ):
            raise ValueError("hand trajectories must be 2D and aligned with timestamps_s")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("hand trajectories contain NaN or Inf")
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "left_joint_positions", left)
        object.__setattr__(self, "right_joint_positions", right)


@dataclass(frozen=True)
class MotionPlan:
    """One VLA prediction chunk plus optional hand targets and provenance."""

    motion: MotionTrajectoryChunk
    hands: HandTrajectory | None = None
    source: str = "vla"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hands is not None and not np.array_equal(
            self.motion.timestamps_s, self.hands.timestamps_s
        ):
            raise ValueError("motion and hand timestamps must be identical")
        object.__setattr__(self, "metadata", dict(self.metadata))
