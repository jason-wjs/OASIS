# Visuomotor policy support

This module ports the Rectified Flow VLA implementation from the local ZSVLA
snapshot into OASIS without importing its stale scene or controller stack.

## Boundaries

- The public deployment output is a controller-neutral MotionPlan defined in
  tasks.teleop.control. The private 69D representation does not leak into
  Teleopit, HEFT, or SONIC adapters.
- CLIP and DINOv2 must be supplied as local directories. Model construction,
  training, caching, and inference never fall back to a network download.
- The known pick-up-basket checkpoint uses action=69, history=2, future=32,
  cameras=3, model_dim=512, and eight denoiser layers. VLAModelConfig records
  this contract; newly trained checkpoints embed the same metadata.
- The original ZSVLA deploy_sim.py is intentionally not copied. It constructs
  an obsolete VAE backbone, depends on a ZSVLA-only low-level controller, and
  references scenes that do not exist in current OASIS. Isaac Lab rollout will
  be wired to the selected production controller on the RTX 4090 feature
  branch.

## Module map

- features.py owns the canonical 69D encoder, online history builder, and
  MotionPlan decoder.
- data.py indexes recorded and visually augmented episodes for training.
- model.py contains the Rectified Flow multimodal backbone and supports injected
  fake encoders for asset-free architecture tests.
- policy.py loads PyTorch checkpoints, preprocesses the three camera streams,
  and provides a latest-observation-wins asynchronous runner.
- train.py and cache_features.py are explicit-path command-line entry points.
- export_inference_ckpt.py and convert_to_inference_ckpt.py preserve model
  architecture metadata while removing training-only state.

## Offline verification

Use the pre-existing ZSVLA environment without installing or changing it:

    /data_team/yzh/miniconda/envs/zsvla/bin/python -m unittest discover -s tests -v

The suite checks the current codec against a real episode and the legacy source
when that snapshot is mounted. It also tests dataset schema, image ordering,
checkpoint shapes, encoder injection, MotionPlan decoding, and async failure
handling.

## Training entry point

The training command requires explicit data and local encoder paths:

    python -m tasks.vla.train \
      --data_root /path/to/episodes \
      --clip_model_path /path/to/local/clip/snapshot \
      --dinov2_repo_path /path/to/local/dinov2 \
      --output_root /path/to/logs

The copied trajectory-only pick-up-basket data is sufficient for codec,
normalization, and motion-data checks. VLA training still requires the matching
camera directories or a compatible precomputed feature cache.
