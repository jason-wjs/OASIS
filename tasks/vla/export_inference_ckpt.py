#!/usr/bin/env python3
"""Export a minimal inference checkpoint from a Stage 2 training checkpoint.

Removes:
  - optimizer state (not needed for inference)
  - VAE encoder weights (only decoder is used at inference)
  - CLIP / DINOv2 weights (loaded from pretrained at init, frozen during training)

Usage:
    python export_inference_ckpt.py \
        --input  ../../assets/ckpts/vla/zsvla_ckp/infer_ckp/stage2_epoch1500.pt \
        --output ../../assets/ckpts/vla/zsvla_ckp/infer_ckp/stage2_inference.pt
"""

import argparse
import os

import torch


# VAE encoder-only keys (not used during inference).
# Shared layers (state_embed, pos_enc) are kept because the decoder needs them.
VAE_ENCODER_PREFIXES = (
    'vae.encoder.',       # TransformerEncoder (encode path)
    'vae.enc_mu.',        # mu projection
    'vae.enc_logvar.',    # logvar projection
    'vae.global_tokens',  # learnable [mu_tok, logvar_tok]
)

# Frozen pretrained encoders — reloaded from HuggingFace / torch.hub at init.
PRETRAINED_PREFIXES = (
    'clip_text_model.',
    'dinov2.',
)

SKIP_PREFIXES = VAE_ENCODER_PREFIXES + PRETRAINED_PREFIXES


def export(input_path: str, output_path: str) -> None:
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')

    full_state = ckpt['model']
    kept = {}
    removed = {}

    for key, val in full_state.items():
        if any(key.startswith(prefix) or key == prefix for prefix in SKIP_PREFIXES):
            removed[key] = val.numel()
        else:
            kept[key] = val

    # Preserve training metadata (useful for provenance)
    out = {
        'model': kept,
        'epoch': ckpt.get('epoch'),
        'loss': ckpt.get('loss'),
        'source': os.path.basename(input_path),
        'model_config': ckpt.get('model_config'),
    }

    torch.save(out, output_path)

    kept_params = sum(v.numel() for v in kept.values())
    removed_params = sum(removed.values())
    total_params = kept_params + removed_params

    orig_mb = os.path.getsize(input_path) / 1024 / 1024
    out_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"  Original : {orig_mb:>8.1f} MB  ({total_params:>12,} params)")
    print(f"  Exported : {out_mb:>8.1f} MB  ({kept_params:>12,} params)")
    print(f"  Removed  : {orig_mb - out_mb:>8.1f} MB  ({removed_params:>12,} params)")
    print(f"  Reduction: {(1 - out_mb / orig_mb) * 100:.1f}%")
    print(f"{'='*60}")

    print(f"\nRemoved keys ({len(removed)}):")
    for key in sorted(removed):
        print(f"  - {key}  ({removed[key]:,} params)")

    print(f"\nKept keys ({len(kept)}):")
    for key in sorted(kept):
        print(f"  + {key}  ({kept[key].numel():,} params)")

    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export minimal inference checkpoint')
    parser.add_argument('--input', required=True, help='Path to Stage 2 training checkpoint')
    parser.add_argument('--output', required=True, help='Output path for inference checkpoint')
    args = parser.parse_args()
    export(args.input, args.output)
