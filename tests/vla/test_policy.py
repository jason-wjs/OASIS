from __future__ import annotations

import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from tasks.params import DEFAULT_HAND_POSE
from tasks.vla.features import HEIGHT_SLICE, FeatureDecoder
from tasks.vla.model import VLABackbone
from tasks.vla.policy import (
    AsyncVLARunner,
    VLAModelConfig,
    preprocess_images,
)


PICK_UP_CHECKPOINT = Path(
    "/data_team/yzh/zsvla/logs/"
    "zsvla_pick_up_basket_2026-05-14_06-35-17/"
    "checkpoints/inference_step060000.pt"
)


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({
            key: value.to(device) for key, value in self.items()
        })


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return TokenBatch({
            "input_ids": torch.ones((len(text), 4), dtype=torch.long)
        })


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)

    def forward(self, input_ids):
        return SimpleNamespace(
            pooler_output=torch.ones(
                (input_ids.shape[0], 8), device=input_ids.device
            )
        )


class FakeDino(nn.Module):
    embed_dim = 12

    def forward_features(self, images):
        return {
            "x_norm_patchtokens": torch.zeros(
                (images.shape[0], 256, self.embed_dim), device=images.device
            )
        }


class FakePolicy:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def infer(self, state_history, images):
        if self.fail:
            raise RuntimeError("expected inference failure")
        result = np.zeros((3, 69), dtype=np.float32)
        result[:, HEIGHT_SLICE] = 0.8
        return result


def _decoder() -> FeatureDecoder:
    hand = DEFAULT_HAND_POSE["unitree_g1"]
    return FeatureDecoder(
        norm_stats={
            "state_mean": np.zeros(69, dtype=np.float32),
            "state_std": np.ones(69, dtype=np.float32),
        },
        hand_open_left=hand["left"]["open"],
        hand_close_left=hand["left"]["close"],
        hand_open_right=hand["right"]["open"],
        hand_close_right=hand["right"]["close"],
    )


class PolicyTest(unittest.TestCase):
    def test_preprocess_images_respects_order_and_strips_alpha(self) -> None:
        images = {
            "head": np.full((10, 12, 3), 10, dtype=np.uint8),
            "left": np.full((10, 12, 4), [20, 20, 20, 255], dtype=np.uint8),
            "right": np.full((10, 12, 3), 30, dtype=np.uint8),
        }
        result = preprocess_images(images)
        self.assertEqual(tuple(result.shape), (1, 3, 3, 224, 224))
        self.assertLess(result[0, 0].mean().item(), result[0, 1].mean().item())
        self.assertLess(result[0, 1].mean().item(), result[0, 2].mean().item())

    def test_injected_backbone_runs_without_pretrained_assets(self) -> None:
        model = VLABackbone(
            action_dim=69,
            history_len=2,
            future_len=3,
            n_cams=3,
            model_dim=16,
            denoiser_layers=1,
            denoiser_heads=4,
            denoiser_ff_size=32,
            num_inference_steps=1,
            dropout=0.0,
            freeze_encoders=True,
            clip_tokenizer=FakeTokenizer(),
            clip_text_model=FakeClip(),
            dinov2=FakeDino(),
        )
        output = model.sample(
            ["pick up basket"],
            torch.zeros((1, 3, 3, 224, 224)),
            torch.zeros((1, 2, 69)),
        )
        self.assertEqual(tuple(output.shape), (1, 3, 69))
        self.assertEqual(tuple(model.cam_emb.shape), (3, 16))
        self.assertEqual(tuple(model.vision_pos_emb.shape), (64, 16))

    @unittest.skipUnless(PICK_UP_CHECKPOINT.is_file(), "pick-up checkpoint is not mounted")
    def test_known_checkpoint_shapes_match_declared_config(self) -> None:
        checkpoint = torch.load(PICK_UP_CHECKPOINT, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        config = VLAModelConfig()
        self.assertEqual(tuple(state["cam_emb"].shape), (config.n_cams, config.model_dim))
        self.assertEqual(
            tuple(state["state_pos_emb"].shape),
            (config.history_len, config.model_dim),
        )
        self.assertEqual(
            tuple(state["denoiser.action_pos_emb"].shape),
            (config.future_len, config.model_dim),
        )
        decoder_layers = {
            int(key.split(".")[3])
            for key in state
            if key.startswith("denoiser.decoder.layers.")
        }
        self.assertEqual(len(decoder_layers), config.denoiser_layers)

    def test_async_runner_publishes_complete_plan_and_stops(self) -> None:
        runner = AsyncVLARunner(FakePolicy(), _decoder())
        try:
            runner.submit(
                np.zeros((2, 69), dtype=np.float32),
                {"head": np.zeros((2, 2, 3), dtype=np.uint8)},
                np.array([0.0, 0.0, 0.8]),
                0.0,
                start_time_s=5.0,
            )
            plan = runner.wait_for_plan(timeout_s=2.0)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.motion.frame_count, 3)
            np.testing.assert_allclose(
                plan.motion.timestamps_s, [5.02, 5.04, 5.06]
            )
            self.assertIsNone(runner.poll_plan())
            self.assertEqual(runner.get_rollout_history(2).shape, (2, 69))
        finally:
            runner.stop()
        self.assertFalse(runner.is_healthy())

    def test_async_runner_reports_fatal_failure(self) -> None:
        runner = AsyncVLARunner(
            FakePolicy(fail=True), _decoder(), max_consecutive_failures=1
        )
        try:
            runner.submit(
                np.zeros((2, 69), dtype=np.float32),
                {},
                np.array([0.0, 0.0, 0.8]),
                0.0,
            )
            self.assertIsNone(runner.wait_for_plan(timeout_s=2.0))
            deadline = time.monotonic() + 2.0
            while runner.latest_error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsInstance(runner.latest_error, RuntimeError)
            self.assertFalse(runner.is_healthy())
        finally:
            runner.stop()


if __name__ == "__main__":
    unittest.main()
