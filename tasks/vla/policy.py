"""Offline-safe PyTorch VLA checkpoint loading and asynchronous inference."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tasks.teleop.control import MotionPlan
from tasks.vla.features import FeatureDecoder, STATE_DIM
from tasks.vla.model import VLABackbone


DEFAULT_CAMERA_ORDER = ("head", "left", "right")
_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@dataclass(frozen=True)
class VLAModelConfig:
    """Architecture values for the pick-up-basket Rectified Flow checkpoint."""

    action_dim: int = 69
    history_len: int = 2
    future_len: int = 32
    n_cams: int = 3
    model_dim: int = 512
    denoiser_layers: int = 8
    denoiser_heads: int = 8
    denoiser_ff_size: int = 2048
    dropout: float = 0.1
    num_inference_steps: int = 10
    dinov2_variant: str = "dinov2_vitb14"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "VLAModelConfig":
        raw = value.get("cfg", value)
        if not isinstance(raw, Mapping):
            raise ValueError("model config must be a mapping or contain a cfg mapping")
        allowed = {field.name for field in fields(cls)}
        selected = {key: raw[key] for key in allowed if key in raw}
        config = cls(**selected)
        if config.action_dim != STATE_DIM:
            raise ValueError(f"VLA action_dim must be {STATE_DIM}, got {config.action_dim}")
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "VLAModelConfig":
        config_path = Path(path).expanduser()
        with config_path.open(encoding="utf-8") as stream:
            return cls.from_mapping(json.load(stream))

    def backbone_kwargs(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("dinov2_variant")
        return result


def preprocess_images(
    images: Mapping[str, np.ndarray],
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
) -> torch.Tensor:
    """Convert RGB/RGBA camera arrays to a (1, cameras, 3, 224, 224) tensor."""
    tensors = []
    for camera_name in camera_order:
        if camera_name not in images:
            raise KeyError(f"missing camera image: {camera_name}")
        image = np.asarray(images[camera_name])
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(
                f"camera {camera_name} must have shape (H, W, 3|4), got {image.shape}"
            )
        if image.shape[-1] == 4:
            image = image[..., :3]
        tensors.append(_IMG_TRANSFORM(Image.fromarray(image.astype(np.uint8))))
    return torch.stack(tensors).unsqueeze(0)


class InferencePolicy(Protocol):
    def infer(
        self,
        state_history: np.ndarray,
        images: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        ...


class VLAPolicy:
    """Load an existing Rectified Flow VLA checkpoint using local encoders only."""

    def __init__(
        self,
        model_path: str | Path,
        norm_stats_path: str | Path,
        model_config: VLAModelConfig | Mapping[str, object] | str | Path,
        clip_model_path: str | Path,
        dinov2_repo_path: str | Path,
        device: str = "cuda",
        text_instruction: str = "pick up the basket",
        camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
        compile_model: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.text = [text_instruction]
        self.camera_order = tuple(camera_order)
        self.config = self._coerce_config(model_config)
        if len(self.camera_order) != self.config.n_cams:
            raise ValueError(
                f"camera_order has {len(self.camera_order)} entries; "
                f"checkpoint expects {self.config.n_cams}"
            )

        stats = np.load(Path(norm_stats_path).expanduser())
        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.maximum(
            np.asarray(stats["state_std"], dtype=np.float32), 1e-8
        )
        if self.state_mean.shape != (STATE_DIM,) or self.state_std.shape != (STATE_DIM,):
            raise ValueError("normalization statistics must both have shape (69,)")

        self._model = VLABackbone(
            **self.config.backbone_kwargs(),
            clip_model_name=str(Path(clip_model_path).expanduser()),
            dinov2_model_name=str(Path(dinov2_repo_path).expanduser()),
            dinov2_variant=self.config.dinov2_variant,
            freeze_encoders=True,
        ).to(self.device)
        checkpoint = torch.load(
            Path(model_path).expanduser(), map_location=self.device, weights_only=True
        )
        if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
            raise ValueError("checkpoint must be a mapping containing a model state dict")
        incompat = self._model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = ("clip_text_model.", "dinov2.")
        bad_missing = [
            key for key in incompat.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if bad_missing or incompat.unexpected_keys:
            raise RuntimeError(
                "checkpoint architecture mismatch: "
                f"missing={bad_missing}, unexpected={incompat.unexpected_keys}"
            )
        self._model.eval()
        if compile_model:
            self._model = torch.compile(self._model, mode="default")

    @staticmethod
    def _coerce_config(
        value: VLAModelConfig | Mapping[str, object] | str | Path,
    ) -> VLAModelConfig:
        if isinstance(value, VLAModelConfig):
            return value
        if isinstance(value, (str, Path)):
            return VLAModelConfig.from_json(value)
        return VLAModelConfig.from_mapping(value)

    def normalize_history(self, state_history: np.ndarray) -> np.ndarray:
        history = np.asarray(state_history, dtype=np.float32)
        expected = (self.config.history_len, STATE_DIM)
        if history.shape != expected:
            raise ValueError(f"state_history must have shape {expected}, got {history.shape}")
        return (history - self.state_mean) / self.state_std

    @torch.no_grad()
    def infer(
        self,
        state_history: np.ndarray,
        images: Mapping[str, np.ndarray],
        *,
        normalized: bool = True,
    ) -> np.ndarray:
        history = np.asarray(state_history, dtype=np.float32)
        expected = (self.config.history_len, STATE_DIM)
        if history.shape != expected:
            raise ValueError(f"state_history must have shape {expected}, got {history.shape}")
        if not normalized:
            history = self.normalize_history(history)
        image_tensor = preprocess_images(images, self.camera_order).to(self.device)
        history_tensor = torch.from_numpy(history).unsqueeze(0).to(self.device)
        future = self._model.sample(self.text, image_tensor, history_tensor)
        result = future.detach().cpu().numpy()[0]
        if result.shape != (self.config.future_len, STATE_DIM):
            raise RuntimeError(f"unexpected model output shape: {result.shape}")
        if not np.isfinite(result).all():
            raise RuntimeError("VLA inference produced NaN or Inf")
        return result


class AsyncVLARunner:
    """Latest-observation-wins inference worker that publishes complete plans."""

    def __init__(
        self,
        policy: InferencePolicy,
        decoder: FeatureDecoder,
        *,
        max_consecutive_failures: int = 10,
    ) -> None:
        self.policy = policy
        self.decoder = decoder
        self.max_consecutive_failures = max_consecutive_failures
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, float, float] | None = None
        self._latest_plan: MotionPlan | None = None
        self._latest_features: np.ndarray | None = None
        self._latest_error: Exception | None = None
        self._prediction_time = 0.0
        self._consecutive_failures = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="oasis-vla-inference", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        history: np.ndarray,
        images: Mapping[str, np.ndarray],
        current_pos: np.ndarray,
        current_yaw: float,
        start_time_s: float = 0.0,
    ) -> None:
        observation = (
            np.asarray(history, dtype=np.float32).copy(),
            {key: np.asarray(value).copy() for key, value in images.items()},
            np.asarray(current_pos, dtype=np.float64).copy(),
            float(current_yaw),
            float(start_time_s),
        )
        with self._condition:
            if not self._running:
                raise RuntimeError("AsyncVLARunner is stopped")
            self._pending = observation
            self._condition.notify()

    def _loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or not self._running
                )
                if not self._running:
                    return
                history, images, position, yaw, start_time_s = self._pending
                self._pending = None
            try:
                features = self.policy.infer(history, images)
                if not np.isfinite(features).all():
                    raise RuntimeError("VLA inference produced NaN or Inf")
                plan = self.decoder.decode_sequence(
                    features,
                    position,
                    yaw,
                    start_time_s=start_time_s,
                )
                with self._condition:
                    self._latest_features = features.copy()
                    self._latest_plan = plan
                    self._prediction_time = time.monotonic()
                    self._latest_error = None
                    self._consecutive_failures = 0
                    self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    self._latest_error = error
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self.max_consecutive_failures:
                        self._running = False
                    self._condition.notify_all()

    def poll_plan(self) -> MotionPlan | None:
        """Return the newest complete plan once, or None when none is pending."""
        with self._condition:
            plan = self._latest_plan
            self._latest_plan = None
            return plan

    def wait_for_plan(self, timeout_s: float) -> MotionPlan | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest_plan is not None
                or self._latest_error is not None
                or not self._running,
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if self._latest_plan is None:
                return None
            plan = self._latest_plan
            self._latest_plan = None
            return plan

    def get_rollout_history(self, history_len: int = 2) -> np.ndarray | None:
        with self._condition:
            if self._latest_features is None:
                return None
            return self._latest_features[-history_len:].copy()

    def is_busy(self) -> bool:
        with self._condition:
            return self._pending is not None

    def is_healthy(self) -> bool:
        with self._condition:
            return self._running and self._thread.is_alive()

    @property
    def latest_error(self) -> Exception | None:
        with self._condition:
            return self._latest_error

    def seconds_since_last_prediction(self) -> float:
        with self._condition:
            if self._prediction_time == 0.0:
                return float("inf")
            return time.monotonic() - self._prediction_time

    def stop(self, timeout_s: float = 2.0) -> None:
        with self._condition:
            self._running = False
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("VLA inference thread did not stop")
