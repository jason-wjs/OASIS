"""Controller-neutral trajectory contracts shared by VLA and teleoperation."""

from .domain import HandTrajectory, MotionPlan, MotionTrajectoryChunk

__all__ = ["HandTrajectory", "MotionPlan", "MotionTrajectoryChunk"]
