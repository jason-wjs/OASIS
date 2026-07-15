# OASIS Multi-Controller and VLA Migration Handoff

Date: 2026-07-15

## Purpose

This document hands the current OASIS controller and visuomotor-policy work to a
fresh session before development moves from the H100 cluster to an RTX 4090
workstation. It records decisions, verified source locations, validation limits,
and the next implementation steps. It contains no credentials or host-specific
service secrets.

## Repository baseline

- Repository: `/data_team/junsong/model-based/OASIS`
- Baseline branch: `main`
- Baseline commit: `f92a6925fd8a87efff3bd6201cd5f1d47d2f2d2f`
- Multi-controller implementation branch to create on the RTX 4090:
  `feat/wujs_multicontroller`
- Do not push or merge the feature branch automatically. The user will merge it
  after rollout validation is stable.

At the time this handoff was first written, `main` was clean. The VLA port and
trajectory copy described below had not yet been implemented.

## Product decisions already made

1. Low-level controller selection happens only at process startup. Runtime hot
   switching is explicitly out of scope.
2. The existing Teleopit controller remains supported.
3. Add HEFT and SONIC as independent low-level motion-tracking controllers.
4. SONIC V1 supports only Unitree G1 full-body trajectory mode. SONIC teleop and
   SMPL modes are out of scope.
5. Controller-specific default joint pose, gains, scaling, and actuator profile
   must be selected before the Isaac Lab scene is constructed.
6. A controller produces targets for the 29 body joints. Dex3 hand commands
   remain on the shared hand-control path.
7. Controller checkpoints may be copied to local `assets/ckpts`, but new binary
   files are ignored by Git. Tracked manifests must validate paths, sizes,
   hashes, joint order, and named ONNX inputs/outputs.
8. Isaac Sim and camera rollout acceptance must happen on RTX hardware. H100
   tests are limited to model inference, pure numerical parity, data handling,
   and non-RTX simulation where available.

## Proposed controller seam

The public controller interface should remain small and hide controller-specific
history, cursor, observation, and post-processing rules:

```python
class MotionTrackingController(Protocol):
    spec: ControllerSpec

    def reset(self, initial_state: RobotState) -> None: ...

    def step(
        self,
        state: RobotState,
        reference_update: MotionTrajectoryChunk | None,
    ) -> BodyJointCommand: ...

    def close(self) -> None: ...
```

Stable domain objects:

- `RobotState`
- `MotionTrajectoryChunk`
- `HandTrajectory`
- `BodyJointCommand`
- `ControllerSpec`
- `RobotControlProfile`

The trajectory object must be name-based and timestamped. Its required body
fields are joint names, 29-DOF joint positions, optional joint velocities, root
position, and root quaternion in documented `wxyz` convention.

## Reference transport decision

The current teleop producer uses a Redis `SET`, so consumers see only the latest
frame. HEFT and SONIC need future windows and must not silently lose intermediate
frames. The planned production adapter is a Redis Stream with sequence and
timestamp fields. An in-memory/recorded-trajectory adapter must satisfy the same
`ReferenceSource` interface for tests and offline rollout.

Reference requirements:

- Teleopit: current/latest reference semantics.
- HEFT: offsets `[-16, ..., +6]`, with a 23-frame temporal span.
- SONIC G1: current through `+9` frames for the encoder window.
- Shared reference code resamples frames onto the 20 ms controller clock and
  reports underflow, stale age, and out-of-order frames explicitly.

## HEFT verified source of truth

- Training repository:
  `/data_team/junsong/tracking/motion_tracking`
- Deployment repository:
  `/data_team/junsong/tracking/deploy/motion_tracking`
- Deployment config:
  `sim2real/config/g1/tracking_wujs.yaml`
- Canonical deploy bundle:
  `sim2real/config/g1/ckpts/wujs/`
- Training export with byte-identical weights:
  `/data_team/junsong/tracking/motion_tracking/assets/ckpts/wjs_v2_localresume_0710/deploy/`

The previously mentioned
`/data_team/junsong/general_controller/motion_tracking/assets/...` path does not
exist on this machine.

HEFT model facts:

- Input: named `policy`, shape `[1, 1729]`.
- Action is the named output `action`; it is not output index zero.
- The 1729 dimensions contain boot state, 13-frame reference features, nine
  robot-state histories, and eight previous raw actions.
- Policy output is clipped and scaled, then added to the controller default
  joint pose.
- History reset, boot-indicator update order, and reference cursor ordering must
  match the deployment repository exactly.

HEFT bundle hashes:

- `policy.onnx`:
  `091ce1570083ba9f2341d233ede6d78f4ab4de2476908a9aca54cebd66259b43`
- `policy.onnx.data`:
  `191b62dba2941ae0af03420add9245621e801bf1cc453dd9015216eecd9ba630`
- `policy.json`:
  `329a7f07d59bad894e647f78e54c53868b0e58a625d381d04b27fdf4099e4740`

## SONIC verified source of truth

- Repository:
  `/data_team/junsong/tracking/GR00T-WholeBodyControl`
- Original training checkpoint:
  `low_latency/last.pt`
- Preferred deploy bundle:
  `gear_sonic_deploy/policy/low_latency/`
- Required files:
  `model_encoder.onnx`, `model_decoder.onnx`, and
  `observation_config.yaml`

SONIC G1 model facts:

- Encoder input: 1247.
- Encoder output token: 64.
- Decoder input: 994.
- Decoder action output: 29.
- G1 encoder mode is mode `0` and consumes a ten-frame full-body trajectory
  window.
- The decoder consumes the token plus ten-frame robot histories and last-action
  history.
- Joint mapping, defaults, gains, and action scaling must come from the upstream
  deployment parameter header, not from the current Teleopit G1 profile.

## H100 and RTX 4090 validation split

The current host has eight NVIDIA H100 80GB GPUs. H100 has no RT Cores and is not
a supported Isaac Sim GPU. The current OASIS scene creates three cameras, and
the replay path uses RTX PathTracing/OptiX. Therefore H100 must not be used to
sign off Isaac Sim rollout behavior.

Safe H100 work:

- Configuration and bundle validation.
- Observation/action golden parity.
- PyTorch and ONNX inference where the runtime is installed.
- Reference buffering and controller unit tests.
- MuJoCo or other non-RTX tests.

Required RTX 4090 work:

- Isaac Sim compatibility preflight.
- Controller-specific robot-profile verification in the constructed scene.
- Camera-disabled and camera-enabled closed-loop rollout.
- Live teleop/reference latency tests.
- One representative PathTracing replay regression.

## VLA source snapshot

The original VLA code is a filesystem snapshot, not a Git worktree:

- Snapshot root: `/data_team/yzh/zsvla`
- Training/data code: `tasks/scripts/`
- Deployment code: `tasks/deploy/`
- Legacy low-level controller:
  `tasks/teleop/controller/zsvla_controller.py`

Do not copy the snapshot's entire `tasks/teleop` tree. Its scene factory, camera
utilities, and parameters have diverged from current OASIS. Port only the VLA
behavior into an isolated `tasks/vla` package and patch current modules
surgically.

The VLA predicts a 69-dimensional motion feature:

- 4D roll/pitch encoding
- yaw increment
- local root-position increment
- absolute root height
- 29 body joint positions
- 29 body joint increments
- left and right hand scalars

The VLA module should decode this internal representation to
`MotionTrajectoryChunk + HandTrajectory`. It must not expose the 69D layout to
low-level controller callers.

VLA support scope on `main`:

- Feature codec and normalization.
- Dataset loading and visual preprocessing.
- Feature-cache generation.
- VLA model/training code.
- Inference checkpoint export.
- Synchronous policy wrapper and asynchronous runner.
- Controller-neutral MotionPlan output in tasks.teleop.control.
- Explicit paths/configuration for CLIP, DINOv2, checkpoints, norm stats, scene,
  and language instruction.

Implementation note: the obsolete snapshot deploy_sim.py was not copied. It
constructs an older VAE backbone that cannot load the current Rectified Flow
checkpoint, depends on a ZSVLA-specific low-level controller, and names scenes
that do not exist in current OASIS. The port instead terminates at the shared,
timestamped MotionPlan boundary. On the RTX 4090 branch, that plan will feed
the startup-selected Teleopit, HEFT, or SONIC adapter. ZSVLA must not be
registered as a fourth production controller.

## Existing offline VLA environment

Do not create or modify a new environment on the H100 host. Reuse this interpreter
only for isolated VLA tests:

`/data_team/yzh/miniconda/envs/zsvla/bin/python`

Verified contents:

- Python 3.10.20
- PyTorch 2.7.0 with CUDA 12.8
- TorchVision 0.22.0
- NumPy 2.2.6
- SciPy 1.15.3
- Transformers 5.9.0
- TensorBoard 2.20.0

It does not contain Isaac Lab, Isaac Sim, or ONNX Runtime. Do not install those
on H100. It also differs from OASIS's NumPy 1.26.4 pin, so use it only across the
isolated VLA interface, never as evidence that the complete OASIS environment is
valid.

The original VLA backbone hard-codes `/data/yzh/...` CLIP and DINOv2 locations.
Those paths are stale on this host. The port must require explicit local model
paths and must not initiate an implicit network download.

## Pick-up-basket trajectory migration

Copy only unique non-visual trajectory JSON files. Preserve each `data.json`
byte-for-byte, including image-path metadata, but omit all image payloads.

Sources:

1. Augmented trajectories:
   `/data_team/yzh/zsvla/tasks/teleop/aug_data/0509_pick_up_basket`
   - 50 episodes
   - approximately 762 MiB of `data.json`
2. Real-data trajectories:
   `/data_team/yzh/zsvla/tasks/teleop/real_data/pick_up_basket`
   - 16 episodes
   - approximately 67 MiB of `data.json`

Do not copy `0509_pick_up_basket_no_texture`: its trajectory JSON files are
duplicates of the ordinary augmented set. Do not copy `env_*`, `colors`,
`.done`, TensorBoard logs, or training checkpoints.

Destination:

`data/trajectories/pick_up_basket/`

The top-level `data/` directory is ignored by Git. A tracked manifest under
`docs/manifests/` must record source, destination, file count, total bytes, and
SHA256 for every copied JSON file. A Git clone on the RTX 4090 will not include
the data directory; synchronize it separately and verify it against the
manifest.

Trajectory-only data is suitable for feature-codec, reference, and low-level
controller tests. It is not sufficient to train a visuomotor policy because the
three camera image payloads are intentionally omitted.

The completed migration is recorded in
docs/manifests/pick_up_basket_trajectories.md; authoritative per-file hashes
are in docs/manifests/pick_up_basket_trajectories.sha256.

## Acceptance criteria before migration

- Original and ported 69D feature encoding match on representative augmented and
  real episodes.
- Decoder reconstruction, quaternion convention, joint order, normalization,
  and image camera order have focused tests.
- Async VLA runner is tested with a fake policy for submit/get, stale output,
  inference failure, and shutdown.
- Training/data modules can import without Isaac Lab.
- Missing images and missing model assets produce explicit errors instead of
  network access or silent fallback.
- Source files compile under the existing offline VLA interpreter.
- The trajectory manifest verifies all copied files.

Full model loading, ONNX inference, Isaac Sim startup, cameras, and closed-loop
rollout remain deferred to the RTX 4090.

## Suggested skills for the next session

- `karpathy-guidelines`: keep the multi-controller diff surgical and make
  verification claims explicit.
- `codebase-design`: preserve the controller and VLA trajectory seams.
- `tdd`: use when implementing controller adapters and their golden-parity
  fixtures test-first.
- `diagnosing-bugs`: use for Isaac Sim startup, observation mismatch, or rollout
  instability on the RTX 4090.

## Next session checklist

1. Verify this handoff's final completion-status section against Git history.
2. Synchronize ignored controller checkpoints and trajectory data to the RTX
   4090 and verify hashes.
3. Install the matching Isaac Lab v2.2.0/Isaac Sim environment and run the GPU
   compatibility checker.
4. Create `feat/wujs_multicontroller` from the migrated `main`.
5. Capture Teleopit golden traces before modifying its controller path.
6. Implement shared domain objects, controller bundles/profiles, and reference
   source.
7. Migrate Teleopit with zero regression, then add HEFT and SONIC.
8. Connect VLA `MotionTrajectoryChunk` output to the selected low-level
   controller.
9. Run camera-disabled, camera-enabled, and representative replay acceptance.

## Completion status

- [x] Multi-controller and VLA requirements investigated.
- [x] H100/RTX 4090 responsibility split agreed.
- [x] Existing offline VLA environment identified.
- [x] VLA module ported to current OASIS.
- [x] VLA offline tests passed (10/10).
- [x] Unique pick-up-basket trajectories copied (66 files, 869485988 bytes).
- [x] Trajectory manifest generated and verified with per-file SHA256.
- [ ] RTX 4090 Isaac Sim and rollout validation completed.
