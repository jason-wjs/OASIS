# Pick-up-basket trajectory migration manifest

Date: 2026-07-15

Only non-visual trajectory files were copied. Every destination data.json is
byte-identical to its source; image-path metadata inside the JSON is preserved,
while image payloads and environment directories are intentionally absent.

| Collection | Source | Destination | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| Augmented 0509 | /data_team/yzh/zsvla/tasks/teleop/aug_data/0509_pick_up_basket | data/trajectories/pick_up_basket/augmented_0509 | 50 | 798982569 |
| Real | /data_team/yzh/zsvla/tasks/teleop/real_data/pick_up_basket | data/trajectories/pick_up_basket/real | 16 | 70503419 |
| Total | | | 66 | 869485988 |

Integrity checks:

- Source and destination file counts and aggregate byte sizes match.
- Per-file SHA256 comparisons between source and destination produced no diff.
- The authoritative destination hashes are in
  pick_up_basket_trajectories.sha256, using paths relative to
  data/trajectories/pick_up_basket.
- No destination files other than episode_*/data.json were found.

Excluded from the migration:

- 0509_pick_up_basket_no_texture, whose trajectories duplicate the augmented set.
- env_* camera payloads, colors, .done markers, logs, and checkpoints.

The top-level data directory is ignored by Git. A new clone must synchronize the
trajectory directory separately and validate it with:

    cd data/trajectories/pick_up_basket
    sha256sum -c ../../../docs/manifests/pick_up_basket_trajectories.sha256

The trajectory-only copy supports codec, normalization, reference, and
low-level-controller tests. It is not sufficient for visuomotor training without
the matching three-camera images or compatible cached visual features.
