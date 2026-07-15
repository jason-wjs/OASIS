"""Visuomotor policy training and controller-neutral deployment support.

Heavy PyTorch and vision dependencies are imported lazily so the 69D feature
codec remains usable in data-only tools.
"""

from .features import (
    FeatureBuilder,
    FeatureDecoder,
    HistoryBuffer,
    STATE_DIM,
    encode_motion_features,
)

__all__ = [
    "AsyncVLARunner",
    "FeatureBuilder",
    "FeatureDecoder",
    "HistoryBuffer",
    "STATE_DIM",
    "VLABackbone",
    "VLAModelConfig",
    "VLAPolicy",
    "encode_motion_features",
]


def __getattr__(name: str):
    if name == "VLABackbone":
        from .model import VLABackbone
        return VLABackbone
    if name in {"AsyncVLARunner", "VLAModelConfig", "VLAPolicy"}:
        from .policy import AsyncVLARunner, VLAModelConfig, VLAPolicy
        return {
            "AsyncVLARunner": AsyncVLARunner,
            "VLAModelConfig": VLAModelConfig,
            "VLAPolicy": VLAPolicy,
        }[name]
    raise AttributeError(name)
