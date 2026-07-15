from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tasks.params import DEFAULT_HAND_POSE, mujoco_joint_names
from tasks.vla.data import RealVLADataset, _load_episode
from tasks.vla.features import (
    DP_LOCAL_SLICE,
    DYAW_SLICE,
    HEIGHT_SLICE,
    LH_SLICE,
    PHI_SLICE,
    QPOS_SLICE,
    RH_SLICE,
    FeatureBuilder,
    FeatureDecoder,
    HistoryBuffer,
    encode_motion_features,
    phi_encode,
)


LEGACY_DATA_MODULE = Path("/data_team/yzh/zsvla/tasks/scripts/data.py")
LEGACY_REAL_EPISODE = Path(
    "/data_team/yzh/zsvla/tasks/teleop/real_data/pick_up_basket/episode_0000"
)


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float64,
    )


class FeatureCodecTest(unittest.TestCase):
    def test_online_builder_matches_batch_encoder(self) -> None:
        frame_count = 6
        positions = np.stack([
            np.array([0.1 * index, 0.02 * index, 0.8 + 0.01 * index])
            for index in range(frame_count)
        ])
        quaternions = np.stack([_yaw_quaternion(0.1 * index) for index in range(frame_count)])
        joints = np.arange(frame_count * 29, dtype=np.float32).reshape(frame_count, 29) / 100
        left = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
        right = left[::-1].copy()

        expected, indices = encode_motion_features(
            positions, quaternions, joints, left, right
        )
        self.assertEqual(indices.tolist(), list(range(frame_count - 1)))

        builder = FeatureBuilder()
        actual = []
        for index in range(frame_count):
            feature = builder.update(
                positions[index],
                quaternions[index],
                joints[index],
                left[index],
                right[index],
            )
            if feature is not None:
                actual.append(feature)
        np.testing.assert_allclose(np.stack(actual), expected, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(
        LEGACY_DATA_MODULE.is_file() and LEGACY_REAL_EPISODE.is_dir(),
        "legacy ZSVLA snapshot is not mounted",
    )
    def test_real_episode_matches_legacy_codec(self) -> None:
        spec = importlib.util.spec_from_file_location("legacy_zsvla_data", LEGACY_DATA_MODULE)
        legacy = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(legacy)
        legacy_features, _, _ = legacy._load_episode(str(LEGACY_REAL_EPISODE))
        current_features, _, _ = _load_episode(str(LEGACY_REAL_EPISODE))
        np.testing.assert_allclose(
            current_features, legacy_features, rtol=0.0, atol=1e-6
        )

    def test_decoder_returns_timed_controller_neutral_plan(self) -> None:
        features = np.zeros((2, 69), dtype=np.float32)
        features[:, PHI_SLICE] = phi_encode(0.05, -0.02)
        features[:, DYAW_SLICE] = 0.1
        features[:, DP_LOCAL_SLICE] = [0.1, 0.0, 0.0]
        features[:, HEIGHT_SLICE] = 0.82
        features[0, QPOS_SLICE] = np.arange(29, dtype=np.float32) / 100
        features[1, QPOS_SLICE] = np.arange(29, dtype=np.float32) / 50
        features[:, LH_SLICE] = [[0.0], [1.0]]
        features[:, RH_SLICE] = [[1.0], [0.0]]

        hand = DEFAULT_HAND_POSE["unitree_g1"]
        decoder = FeatureDecoder(
            norm_stats={
                "state_mean": np.zeros(69, dtype=np.float32),
                "state_std": np.ones(69, dtype=np.float32),
            },
            hand_open_left=hand["left"]["open"],
            hand_close_left=hand["left"]["close"],
            hand_open_right=hand["right"]["open"],
            hand_close_right=hand["right"]["close"],
        )
        plan = decoder.decode_sequence(
            features,
            current_pos=np.array([1.0, 2.0, 0.8]),
            current_yaw=0.0,
            start_time_s=10.0,
        )

        self.assertEqual(plan.motion.joint_names, tuple(mujoco_joint_names))
        np.testing.assert_allclose(plan.motion.timestamps_s, [10.02, 10.04])
        np.testing.assert_allclose(plan.motion.root_positions[0], [1.1, 2.0, 0.82])
        np.testing.assert_allclose(
            plan.hands.left_joint_positions[0], hand["left"]["open"]
        )
        np.testing.assert_allclose(
            plan.hands.left_joint_positions[1], hand["left"]["close"]
        )
        np.testing.assert_allclose(
            np.linalg.norm(plan.motion.root_quaternions_wxyz, axis=1), 1.0
        )

    def test_history_buffer_left_pads_first_frame(self) -> None:
        history = HistoryBuffer(maxlen=2)
        history.append(np.arange(69, dtype=np.float32))
        padded = history.get_padded()
        self.assertEqual(padded.shape, (2, 69))
        np.testing.assert_array_equal(padded[0], padded[1])

    def test_dataset_reads_current_episode_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_0000"
            colors = episode / "env_0" / "colors"
            colors.mkdir(parents=True)
            frames = []
            for frame_index in range(7):
                color_paths = {}
                for camera_index in range(3):
                    filename = f"{frame_index:06d}_color_{camera_index}.png"
                    Image.new("RGB", (8, 8), (frame_index, camera_index, 10)).save(
                        colors / filename
                    )
                    color_paths[f"color_{camera_index}"] = f"colors/{filename}"
                frames.append({
                    "action_mimic": {
                        "body": {
                            "root_pos": [0.01 * frame_index, 0.0, 0.8],
                            "root_quat": _yaw_quaternion(0.01 * frame_index).tolist(),
                            "joint_pos": (np.ones(29) * frame_index / 100).tolist(),
                        },
                        "left_hand": {"qpos": 0.0},
                        "right_hand": {"qpos": 1.0},
                    },
                    "colors": color_paths,
                })
            with (episode / "data.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {"info": {"scene": "table_basket"}, "text": "pick up basket", "data": frames},
                    stream,
                )

            dataset = RealVLADataset(
                str(root), history_len=2, future_len=3, image_size=16
            )
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["state_history"].shape), (2, 69))
            self.assertEqual(tuple(sample["future_state"].shape), (3, 69))
            self.assertEqual(tuple(sample["image"].shape), (3, 3, 16, 16))
            self.assertEqual(sample["text"], "pick up basket")


if __name__ == "__main__":
    unittest.main()
