"""
Pre-compute frozen encoder features (DINOv2 + CLIP) for VLA training.
===================================================================

Caches per-env DINOv2 *patch* tokens and CLIP text embeddings so that
flow-matching training can skip the expensive frozen-encoder forward pass.
The cached tensors match the shapes expected by
``VLABackbone._encode_condition_cached`` and are consumed by
``RolloutPrimitiveDataset`` when ``cache_dir`` is provided.

Cache layout
------------
  <cache_dir>/
    <episode_name>__<env_name>.pt
      → dict with keys:
          'vision_patches': Tensor (N_frames, n_cams, N_patches, dinov2_dim)
          'text_feat':      Tensor (clip_dim,)
          'img_root':       str   — original env image root (for traceability)

  For DINOv2 ViT-B/14 at 224×224 input:
      N_patches = (224 / 14)² = 16 × 16 = 256
      dinov2_dim = 768

Usage
-----
    python cache_features.py --data_root ../teleop/aug_data --cache_dir ../teleop/aug_data_cache
    python cache_features.py --data_root ../teleop/aug_data  # default: <data_root>_cache

Disk footprint warning
----------------------
Patch tokens are ~256× larger than a single CLS token. Rough estimates
(float32):
  * per image (256 × 768 × 4B)        ≈ 0.77 MB
  * per 3-cam frame                    ≈ 2.3 MB
  * per 1000-frame episode (3 cams)    ≈ 2.3 GB
Make sure the target disk has enough capacity before running.
"""

import os
import argparse
import json

from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from tasks.params import VLA_STRIDE


# ── Image transform (must match training) ────────────────────────────────────

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]


def _build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    ])


# ── Episode / env discovery ──────────────────────────────────────────────────

def _sorted_episode_dirs(data_root: str):
    return sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if d.startswith('episode_') and os.path.isdir(os.path.join(data_root, d))
    ])


def _discover_envs(ep_dir: str):
    """
    Discover all env directories inside an episode and the corresponding
    image frame indices (subsampled by VLA_STRIDE).

    Returns list of dicts:
        env_name    : str
        img_root    : str                   absolute path to env image root
        frame_paths : list[list[str]]       N_frames × n_cams absolute paths
        text        : str
    """
    stride = VLA_STRIDE

    with open(os.path.join(ep_dir, 'data.json')) as f:
        data = json.load(f)

    frames = data['data']
    T      = len(frames)

    text_raw = data.get('text', {})
    if isinstance(text_raw, str) and text_raw.strip():
        text = text_raw.strip()
    else:
        info = data.get('info', {})
        scene = info.get('scene', '') if isinstance(info, dict) else ''
        text = str(scene).strip() or "pick up the basket"

    n_cams = len(frames[0]['colors'])
    sample_indices = list(range(0, T - stride, stride))

    # Current data layout: ep_dir/env_0/colors/XXXXXX_color_Y.jpg
    # Discover env dirs that contain a 'colors' subdirectory.
    env_dirs = sorted([
        d for d in os.listdir(ep_dir)
        if d.startswith('env_') and os.path.isdir(os.path.join(ep_dir, d, 'colors'))
    ])

    # Fallback: if no env dirs found, use the default 'env_0'.
    if not env_dirs:
        env_dirs = ['env_0']

    results = []
    for env_name in env_dirs:
        img_root = os.path.join(ep_dir, env_name)
        frame_paths = []
        for t in sample_indices:
            cam_paths = [
                os.path.join(img_root, frames[t]['colors'][f'color_{c}'])
                for c in range(n_cams)
            ]
            if all(os.path.isfile(p) for p in cam_paths):
                frame_paths.append(cam_paths)
            else:
                break  # Incomplete image sequence for this env — stop.

        if len(frame_paths) > 0:
            results.append({
                'env_name':    env_name,
                'img_root':    img_root,
                'frame_paths': frame_paths,
                'text':        text,
            })

    return results


def _cache_key(ep_dir: str, env_name: str) -> str:
    """Generate cache filename from episode dir and env name."""
    ep_name = os.path.basename(ep_dir)
    return f"{ep_name}__{env_name}.pt"


# ── Main caching logic ───────────────────────────────────────────────────────

@torch.no_grad()
def cache_all(
    data_root: str,
    cache_dir: str,
    clip_model_path: str,
    dinov2_repo_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    dinov2_variant: str = "dinov2_vitb14",
    device: str = "cuda",
):
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = _build_transform(image_size)

    # ── Load encoders ────────────────────────────────────────────────────
    print("Loading CLIP text encoder...")
    from transformers import CLIPTokenizer, CLIPTextModel
    clip_path = Path(clip_model_path).expanduser()
    dinov2_path = Path(dinov2_repo_path).expanduser()
    if not clip_path.is_dir():
        raise FileNotFoundError(f"local CLIP model not found: {clip_path}")
    if not dinov2_path.is_dir():
        raise FileNotFoundError(f"local DINOv2 repo not found: {dinov2_path}")
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        str(clip_path), local_files_only=True
    )
    clip_text_model = CLIPTextModel.from_pretrained(
        str(clip_path), local_files_only=True
    ).to(device).eval()

    print("Loading DINOv2 vision encoder...")
    dinov2 = torch.hub.load(
        str(dinov2_path), dinov2_variant, pretrained=True, source='local'
    ).to(device).eval()

    # ── Discover all (episode, env) pairs ────────────────────────────────
    episode_dirs = _sorted_episode_dirs(data_root)
    all_jobs = []
    for ep_dir in episode_dirs:
        envs = _discover_envs(ep_dir)
        for env_info in envs:
            key = _cache_key(ep_dir, env_info['env_name'])
            cache_path = os.path.join(cache_dir, key)
            if os.path.isfile(cache_path):
                continue  # Already cached.
            all_jobs.append((ep_dir, env_info, cache_path))

    print(f"Found {len(episode_dirs)} episodes, "
          f"{len(all_jobs)} envs to cache (skipping already cached)")

    if not all_jobs:
        print("Nothing to cache — all envs already cached.")
        return

    # ── Cache CLIP text embeddings (unique texts only) ───────────────────
    unique_texts = list({job[1]['text'] for job in all_jobs})
    text_cache   = {}
    for text in unique_texts:
        tokens = clip_tokenizer(
            text, return_tensors='pt', padding=True,
            truncation=True, max_length=77,
        ).to(device)
        text_feat = clip_text_model(**tokens).pooler_output.squeeze(0).cpu()
        text_cache[text] = text_feat                # (clip_dim,)
    print(f"Cached {len(unique_texts)} unique text embeddings")

    # ── Cache DINOv2 patch tokens per env ────────────────────────────────
    for ep_dir, env_info, cache_path in tqdm(all_jobs, desc="Caching envs"):
        frame_paths = env_info['frame_paths']
        n_frames    = len(frame_paths)
        n_cams      = len(frame_paths[0])

        all_patches = []
        # Process in batches of frames to control peak memory.
        for i in range(0, n_frames, batch_size):
            batch_frames = frame_paths[i : i + batch_size]
            # Load all camera images for this batch: (batch * n_cams, 3, H, W)
            imgs = []
            for cam_paths in batch_frames:
                for p in cam_paths:
                    imgs.append(transform(Image.open(p).convert('RGB')))
            imgs_tensor = torch.stack(imgs).to(device)

            patch_tokens = dinov2.forward_features(imgs_tensor)['x_norm_patchtokens']
            #              (batch * n_cams, N_patches, dinov2_dim)
            N_patches, D_vis = patch_tokens.shape[-2], patch_tokens.shape[-1]
            patch_tokens = patch_tokens.reshape(-1, n_cams, N_patches, D_vis)
            #              (batch, n_cams, N_patches, dinov2_dim)
            all_patches.append(patch_tokens.cpu())

        vision_patches = torch.cat(all_patches, dim=0)
        #                (N_frames, n_cams, N_patches, dinov2_dim)

        torch.save({
            'vision_patches': vision_patches,
            'text_feat':      text_cache[env_info['text']],
            'img_root':       env_info['img_root'],
        }, cache_path)

    print(f"Done. Cached {len(all_jobs)} envs to {cache_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute frozen encoder features")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Path to episode data directory")
    parser.add_argument('--clip_model_path', required=True,
                        help="Local Hugging Face CLIP snapshot")
    parser.add_argument('--dinov2_repo_path', required=True,
                        help="Local facebookresearch/dinov2 repository")
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="Output cache directory (default: <data_root>_cache)")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DINOv2 inference")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = args.data_root.rstrip('/') + '_cache'

    cache_all(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        clip_model_path=args.clip_model_path,
        dinov2_repo_path=args.dinov2_repo_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
