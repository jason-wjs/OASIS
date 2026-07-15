"""
VLA Flow-Matching Rollout Training Script
==========================================
Single-stage Rectified Flow Matching training with DP-Transformer denoiser
and autoregressive rollout (scheduled sampling).

Training is step-based (not epoch-based).  Each step processes ``num_primitive``
consecutive primitives from randomly sampled episodes. Scheduled sampling
linearly ramps the rollout probability from 0 to ``p_rollout_max`` after a
warmup phase, so the model gradually learns to handle its own prediction errors.

  Per step (rollout with P primitives):
      For p = 0..P-1:
          history = (pure GT) | (last H of previous prediction, with prob p_rollout)
          loss    = model.flow_matching_loss(text, image, history, gt_future)
          loss.backward(); optim.step()
          if p_rollout > 0 and p < P-1:
              prev_pred = model.sample(text, image, history)   # 10-step Euler

Usage:
    python train.py
    python train.py --resume logs/<run_id>/checkpoints/flow_step030000.pt

Notes:
    * Normalisation is owned by the data pipeline, not the model. data.py
      normalises features at load time; ``norm_stats.npz`` is saved alongside
      checkpoints so deployment can normalise inputs and denormalise the
      model's sampled actions.
    * The ``cache_dir`` path from the previous DART-era pipeline cached CLS
      tokens; the new backbone needs *patch tokens*, so cached training is
      disabled here until the caching script is regenerated.
"""

import os
import math
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tasks.vla.model import VLABackbone
from tasks.vla.data import (
    RolloutPrimitiveDataset,
    compute_norm_stats,
)


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy / random per-worker so different workers don't sample identically."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = dict(
    # Data
    data_root     = None,
    cache_dir     = None,    # set to e.g. "../teleop/aug_data_cache" to skip CLIP+DINOv2 forward
    clip_model_path = None,
    dinov2_repo_path = None,
    state_dim     = 69,
    history_len   = 2,
    future_len    = 32,
    num_primitive = 4,       # rollout depth; segment_len = H + F*P frames
    n_cams        = 3,
    image_size    = 224,
    max_episodes  = None,    # int → randomly sample this many episodes from data_root; None → use all
    episode_seed  = 42,      # RNG seed for episode subsampling (independent of training seed)

    max_envs_per_episode = None,   # int → cap envs per episode; None → use all
    env_seed      = 42, 
    model_dim     = 512,

    # Denoiser (DP-Transformer)
    denoiser_layers   = 8,
    denoiser_heads    = 8,
    denoiser_ff_size  = 2048,

    # Flow matching inference
    num_inference_steps = 10,

    # Training
    flow_steps    = 50000,
    lr            = 1e-5,     # peak LR (after warmup)
    min_lr        = 1e-5,     # LR floor at end of cosine decay
    warmup_steps  = 10000,     # linear warmup from lr/warmup_steps to peak over this many steps
    wd            = 1e-4,

    batch_size = 64,        # sequence is longer now; smaller batch to fit
    num_workers = 4,        # DataLoader workers for async image IO
    dropout    = 0.1,
    save_every = 5000,
    log_every  = 50,

    # Rollout scheduling
    warmup_ratio  = 0.2,
    p_rollout_max = 0.6,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_rollout_dataset(cfg: dict, norm_stats: dict) -> RolloutPrimitiveDataset:
    return RolloutPrimitiveDataset(
        data_root     = cfg['data_root'],
        history_len   = cfg['history_len'],
        future_len    = cfg['future_len'],
        num_primitive = cfg['num_primitive'],
        batch_size    = cfg['batch_size'],
        image_size    = cfg['image_size'],
        norm_stats    = norm_stats,
        cache_dir     = cfg['cache_dir'],   # None → on-the-fly encoding; dir → use cached patch tokens
        max_episodes  = cfg['max_episodes'],
        episode_seed  = cfg['episode_seed'],
        max_envs_per_episode = cfg['max_envs_per_episode'],
        env_seed             = cfg['env_seed'],
    )


# ---------------------------------------------------------------------------
# Rollout scheduling
# ---------------------------------------------------------------------------

# Local ramp (in steps) applied after a resume so p_rollout doesn't jump from
# 0 straight to its "natural" absolute-step value (gradient shock).
_RESUME_RAMP_STEPS = 500


def compute_lr(step: int, cfg: dict) -> float:
    """
    Learning-rate schedule: linear warmup → cosine decay to ``min_lr``.

    Purely a function of the absolute step number, so it behaves identically
    across resumes (no scheduler state to save or reload).

    - step ∈ [1, warmup_steps]:          lr_peak · step / warmup_steps
    - step ∈ (warmup_steps, flow_steps]: cosine from lr_peak down to min_lr
    - step > flow_steps:                 clamped to min_lr
    """
    lr_peak = cfg['lr']
    lr_min  = cfg['min_lr']
    warmup  = cfg['warmup_steps']
    total   = cfg['flow_steps']

    # if step <= warmup:
    #     return lr_peak * step / max(1, warmup)

    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))        # 1 → 0
    #return lr_min + (lr_peak - lr_min) * cosine
    return lr_peak


def get_rollout_prob(step: int, total_steps: int,
                    warmup_ratio: float, p_max: float,
                    start_step: int = 1) -> float:
    """
    Compute rollout probability at a given training step.

    - Global warmup: first ``warmup_ratio`` of total steps → p_rollout = 0.
    - Global ramp: linear 0 → p_max over the remaining steps.
    - Resume ramp: when the run resumes past a completed warmup, p_rollout is
      additionally scaled by a linear 0→1 ramp over the first
      ``_RESUME_RAMP_STEPS`` steps after ``start_step``. For a fresh run
      (start_step = 1) this window lies inside warmup, so it has no effect.
    """
    warmup_steps = int(total_steps * warmup_ratio)
    if step <= warmup_steps:
        return 0.0
    progress  = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    p_rollout = min(p_max, p_max * progress)

    if start_step > 1 and _RESUME_RAMP_STEPS > 0:
        resume_factor = min(1.0, max(0.0, (step - start_step) / _RESUME_RAMP_STEPS))
        p_rollout    *= resume_factor

    return p_rollout


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_FROZEN_ENCODER_PREFIXES = (
    'clip_text_model.', 'dinov2.',   # reloaded from HF / torch.hub at model init
)


def _model_config(cfg: dict) -> dict:
    """Serializable architecture fields required to reconstruct a checkpoint."""
    return {
        'action_dim': cfg['state_dim'],
        'history_len': cfg['history_len'],
        'future_len': cfg['future_len'],
        'n_cams': cfg['n_cams'],
        'model_dim': cfg['model_dim'],
        'denoiser_layers': cfg['denoiser_layers'],
        'denoiser_heads': cfg['denoiser_heads'],
        'denoiser_ff_size': cfg['denoiser_ff_size'],
        'dropout': cfg['dropout'],
        'num_inference_steps': cfg['num_inference_steps'],
        'dinov2_variant': 'dinov2_vitb14',
    }


def _trainable_state_dict(model):
    """Model state_dict minus the frozen pretrained encoders (reloaded at init)."""
    return {
        k: v for k, v in model.state_dict().items()
        if not any(k.startswith(p) for p in _FROZEN_ENCODER_PREFIXES)
    }


def save_checkpoint(model, optimizer, step, loss, tag, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}_step{step:06d}.pt")
    torch.save({
        'step': step, 'loss': loss,
        'model': _trainable_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'model_config': _model_config(CFG),
    }, path)
    print(f"  Saved -> {path}")
    return path


def save_inference_checkpoint(model, step, loss, ckpt_dir):
    """Minimal deployment checkpoint — omit optimiser and frozen pretrained encoders."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state = _trainable_state_dict(model)
    path = os.path.join(ckpt_dir, f"inference_step{step:06d}.pt")
    torch.save({
        'step': step,
        'loss': loss,
        'model': state,
        'model_config': _model_config(CFG),
    }, path)
    print(f"  Saved inference ckpt -> {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    incompat = model.load_state_dict(ckpt['model'], strict=False)
    bad_missing = [
        key for key in incompat.missing_keys
        if not key.startswith(_FROZEN_ENCODER_PREFIXES)
    ]
    if bad_missing or incompat.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={bad_missing}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    optimizer.load_state_dict(ckpt['optimizer'])
    step = ckpt.get('step', 0)
    print(f"  Resumed from {path}  (step {step}, loss {ckpt['loss']:.4f})")
    return step + 1


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def build_optimizer(model: VLABackbone, cfg: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"[optim] trainable params: {n_params:,}")
    return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])


# ---------------------------------------------------------------------------
# Flow-matching rollout training
# ---------------------------------------------------------------------------

def train_flow_rollout(model, dataset, optimizer, cfg, device, writer,
                       ckpt_dir: str, start_step: int = 1):
    """
    Step-based flow-matching training with rollout.

    Each step:
      1. Sample a batch of P consecutive primitives.
      2. For pidx = 0..P-1, optionally use model's own prediction as history
         (scheduled sampling), compute flow-matching loss, backward, step.
    """
    model.train()
    dataiter    = iter(dataset)
    total_steps = cfg['flow_steps']
    H = cfg['history_len']
    P = cfg['num_primitive']

    avg_loss = 0.0
    for step in range(start_step, total_steps + 1):
        batch = next(dataiter)

        # LR schedule: warmup + cosine decay. Applied once at the start of the
        # step, so all P primitives in this iteration share the same LR.
        cur_lr = compute_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        p_rollout = get_rollout_prob(
            step, total_steps, cfg['warmup_ratio'], cfg['p_rollout_max'],
            start_step=start_step)

        prev_pred      = None
        step_loss      = 0.0
        losses_gt      = []     # loss values where history came from ground truth
        losses_rollout = []     # loss values where history came from model self-prediction

        for pidx in range(P):
            prim       = batch[pidx]
            features   = prim['features'].to(device)              # (B, H+F, 69)
            gt_history = features[:, :H, :]
            gt_future  = features[:, H:, :]

            # Scheduled sampling: decide GT vs rollout history for this primitive.
            use_rollout = (
                pidx > 0
                and prev_pred is not None
                and random.random() < p_rollout
            )
            history = prev_pred[:, -H:, :].detach() if use_rollout else gt_history

            # Frozen-encoder outputs: either pull from disk cache (fast path) or
            # run CLIP + DINOv2 on-the-fly. Dataset emits mutually exclusive
            # fields: cache mode → 'vision_patches' + 'text_feat'; no-cache mode
            # → 'images'.
            if 'vision_patches' in prim:
                text_feat      = prim['text_feat'].to(device, non_blocking=True)
                vision_patches = prim['vision_patches'].to(device, non_blocking=True)
            else:
                images         = prim['images'].to(device, non_blocking=True)
                text_feat, vision_patches = model.encode_frozen(prim['text'], images)

            loss, _ = model.flow_matching_loss_cached(
                text_feat, vision_patches, history, gt_future)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            loss_val   = loss.item()
            step_loss += loss_val

            # Per-pidx loss — unique tag per primitive slot so pidx curves can be
            # compared directly (e.g. pidx=1 should trend towards pidx=0 as
            # rollout robustness improves).
            writer.add_scalar(f'flow/loss_pidx{pidx}', loss_val, step)

            # Bucket by history source for downstream GT-vs-rollout analysis.
            (losses_rollout if use_rollout else losses_gt).append(loss_val)

            # Generate prediction for next primitive's rollout (only when needed).
            # Switch to eval so Dropout is off during sampling; sample_cached
            # is already @torch.no_grad(). train() is re-asserted after.
            if p_rollout > 0 and pidx < P - 1:
                model.eval()
                prev_pred = model.sample_cached(text_feat, vision_patches, history)
                model.train()

        avg_loss = step_loss / P

        writer.add_scalar('flow/step_loss', avg_loss,  step)
        writer.add_scalar('flow/p_rollout', p_rollout, step)
        writer.add_scalar('flow/lr',        cur_lr,    step)
        # GT vs rollout loss split — only emit when that bucket actually got a
        # sample this step (during warmup `losses_rollout` stays empty).
        if losses_gt:
            writer.add_scalar('flow/loss_gt',      np.mean(losses_gt),      step)
        if losses_rollout:
            writer.add_scalar('flow/loss_rollout', np.mean(losses_rollout), step)

        if step % cfg['log_every'] == 0:
            print(f"  [Flow] step {step:6d}/{total_steps} | "
                  f"loss {avg_loss:.4f} | p_rollout {p_rollout:.3f} | "
                  f"lr {cur_lr:.2e}",
                  flush=True)

        # if step % cfg['save_every'] == 0:
        #     save_checkpoint(model, optimizer, step, avg_loss,
        #                     tag='flow', ckpt_dir=ckpt_dir)

    return avg_loss


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(device) -> VLABackbone:
    return VLABackbone(
        action_dim          = CFG['state_dim'],
        history_len         = CFG['history_len'],
        future_len          = CFG['future_len'],
        n_cams              = CFG['n_cams'],
        model_dim           = CFG['model_dim'],
        denoiser_layers     = CFG['denoiser_layers'],
        denoiser_heads      = CFG['denoiser_heads'],
        denoiser_ff_size    = CFG['denoiser_ff_size'],
        dropout             = CFG['dropout'],
        num_inference_steps = CFG['num_inference_steps'],
        freeze_encoders     = True,
        clip_model_name     = CFG['clip_model_path'],
        dinov2_model_name   = CFG['dinov2_repo_path'],
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="VLA Flow-Matching Rollout Training")
    parser.add_argument(
        '--data_root', required=True,
        help="Episode root containing episode_*/data.json and camera directories.",
    )
    parser.add_argument(
        '--clip_model_path', required=True,
        help="Local Hugging Face CLIP snapshot; network downloads are disabled.",
    )
    parser.add_argument(
        '--dinov2_repo_path', required=True,
        help="Local facebookresearch/dinov2 torch.hub repository.",
    )
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument(
        '--output_root', default='logs',
        help="Directory for checkpoints, stats, config, and TensorBoard logs.",
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Checkpoint path to resume training from.",
    )
    parser.add_argument(
        '--run_name', type=str, default=None,
        help="Optional human-readable prefix for the run directory: "
             "logs/<run_name>_<timestamp>/. Omit for plain timestamp.",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    CFG.update(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        clip_model_path=args.clip_model_path,
        dinov2_repo_path=args.dinov2_repo_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id     = f"{args.run_name}_{timestamp}" if args.run_name else timestamp
    run_dir    = os.path.join(args.output_root, run_id)
    ckpt_dir   = os.path.join(run_dir, "checkpoints")
    tb_log_dir = os.path.join(run_dir, "runs", "vla")
    print(f"Run ID: {run_id}")

    norm_stats = compute_norm_stats(CFG['data_root'])
    os.makedirs(run_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "norm_stats.npz")
    np.savez(stats_path,
             state_mean=norm_stats['state_mean'],
             state_std =norm_stats['state_std'])
    print(f"  norm_stats saved -> {stats_path}")

    import json
    cfg_path = os.path.join(run_dir, "train_cfg.json")
    with open(cfg_path, 'w') as f:
        json.dump({
            'cfg':       CFG,
        }, f, indent=2)
    print(f"  train_cfg saved -> {cfg_path}")

    dataset = build_rollout_dataset(CFG, norm_stats)
    loader  = DataLoader(
        dataset,
        batch_size         = None,      # dataset already yields batched primitives
        num_workers        = CFG['num_workers'],
        pin_memory         = True,
        prefetch_factor    = 2    if CFG['num_workers'] > 0 else None,
        persistent_workers = True if CFG['num_workers'] > 0 else False,
        worker_init_fn     = _worker_init_fn,
    )

    model = build_model(device)
    # Normalisation is owned by the data pipeline, not the model.
    # data.py normalises features at load time; deploy scripts normalise inputs
    # and denormalise outputs via the ``norm_stats.npz`` saved above. The
    # backbone itself operates purely in normalised 69D space.

    optimizer  = build_optimizer(model, CFG)
    start_step = 1
    if args.resume and os.path.isfile(args.resume):
        start_step = load_checkpoint(model, optimizer, args.resume, device)

    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs -> {tb_log_dir}  "
          f"(tensorboard --logdir logs/{run_id})\n")

    print("=" * 60)
    print("Flow-Matching Rollout Training")
    print("=" * 60)

    avg = train_flow_rollout(
        model, loader, optimizer, CFG, device, writer, ckpt_dir, start_step)

    # save_checkpoint(model, optimizer, CFG['flow_steps'], avg,
    #                 tag='flow_final', ckpt_dir=ckpt_dir)
    save_inference_checkpoint(model, CFG['flow_steps'], avg,
                              ckpt_dir=ckpt_dir)

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
