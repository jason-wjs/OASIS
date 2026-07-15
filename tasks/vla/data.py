"""
VLA Real Data Loader  (Motion-Feature Edition)
===============================================
Per-frame motion feature  f_t ∈ ℝ^69:

  f_t = [φ(r_t),  Δψ_t,  Δp_t^local,  h_t,  q_t,  Δq_t,  lh_t,  rh_t]

  φ(r_t)       [0 :4 ]  sin/cos encoding of roll & pitch          (4D)
  Δψ_t         [4 :5 ]  incremental yaw  yaw_{t+1} - yaw_t       (1D)
  Δp_t^local   [5 :8 ]  root translation increment in yaw-frame   (3D)
  h_t          [8 :9 ]  root height                               (1D)
  q_t          [9 :38]  body joint positions                      (29D)
  Δq_t         [38:67]  body joint-wise increments                (29D)
  lh_t         [67:68]  left  hand gripper action (scalar)        (1D)
  rh_t         [68:69]  right hand gripper action (scalar)        (1D)

  φ(r_t) = [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]
  Δp_t^local = Rz(yaw_t)ᵀ @ (pos_{t+1} - pos_t)

Root quaternion convention: MuJoCo [w, x, y, z].
Last frame of every episode is discarded (requires t+1).
Window:  history_len frames of feature  +  future_len frames of feature.
Images:  3 cameras at the last history frame, resized to image_size×image_size.
Text:    info.scene used as fallback when the text field is empty.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from PIL import Image

from tasks.params import VLA_STRIDE
from tasks.vla.features import STATE_DIM, encode_motion_features


# ── Episode loader ────────────────────────────────────────────────────────────

def _load_episode(ep_dir: str):
    """
    Parse one episode directory and build the motion feature sequence.

    With VLA_STRIDE > 1, features are subsampled: absolute values are taken
    at every stride-th frame, and delta values span stride frames.  This
    increases the effective prediction horizon (stride × 20ms per feature).

    Returns
    -------
    features  : np.ndarray       shape (N, 69), N = (T - stride) // stride
    img_paths : list[list[str]]  length N, each inner list has n_cams paths
    text      : str
    """
    stride = VLA_STRIDE

    with open(os.path.join(ep_dir, 'data.json')) as f:
        data = json.load(f)

    frames = data['data']
    info   = data['info']
    T      = len(frames)

    text_raw = data.get('text', {})
    if isinstance(text_raw, str) and text_raw.strip():
        text = text_raw.strip()
    else:
        scene = info.get('scene', '') if isinstance(info, dict) else ''
        text = str(scene).strip() or "pick up the basket"

    pos   = np.array([fr['action_mimic']['body']['root_pos']  for fr in frames], dtype=np.float64)  # (T, 3)
    quats = np.array([fr['action_mimic']['body']['root_quat'] for fr in frames], dtype=np.float64)  # (T, 4)
    qpos  = np.array([fr['action_mimic']['body']['joint_pos'] for fr in frames], dtype=np.float32)  # (T, 29)
    lh    = np.array([fr['action_mimic']['left_hand']['qpos']  for fr in frames], dtype=np.float32)  # (T,)
    rh    = np.array([fr['action_mimic']['right_hand']['qpos'] for fr in frames], dtype=np.float32)  # (T,)

    features, sample_indices_array = encode_motion_features(
        pos, quats, qpos, lh, rh, stride=stride
    )
    sample_indices = sample_indices_array.tolist()

    n_cams = len(frames[0]['colors'])

    # Discover all env directories that have images
    env_dirs = sorted([
        d for d in os.listdir(ep_dir)
        if d.startswith('env_') and os.path.isdir(os.path.join(ep_dir, d, 'colors'))
    ])
    if not env_dirs:
        env_dirs = ['env_0']

    # Build img_paths per env: dict[env_name] -> list[list[str]]
    envs = {}
    for env_name in env_dirs:
        img_root = os.path.join(ep_dir, env_name)
        paths = [
            [os.path.join(img_root, frames[t]['colors'][f'color_{c}'])
             for c in range(n_cams)]
            for t in sample_indices
        ]
        # Only include env if first frame images exist
        if all(os.path.isfile(p) for p in paths[0]):
            envs[env_name] = paths

    return features, envs, text


# ── Normalisation stats ───────────────────────────────────────────────────────

def compute_norm_stats(data_root: str) -> dict:
    """
    Compute per-dimension mean and std over every valid frame in the dataset.

    Returns
    -------
    dict with keys 'state_mean' and 'state_std', each np.ndarray of shape (69,).
    """
    episode_dirs  = _sorted_episode_dirs(data_root)
    all_features  = []
    for ep_dir in episode_dirs:
        features, _, _ = _load_episode(ep_dir)
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)   # (N_total, 67)
    mean = all_features.mean(axis=0).astype(np.float32)

    std  = np.maximum(all_features.std(axis=0).astype(np.float32), 1e-8)

    mean[67:69] = 0.0
    std [67:69] = 1.0
    
    print(
        f"[compute_norm_stats] {all_features.shape[0]} frames  "
        f"| mean ∈ [{mean.min():.4f}, {mean.max():.4f}]  "
        f"| std  ∈ [{std.min():.4f},  {std.max():.4f}]"
    )
    return {'state_mean': mean, 'state_std': std}


# ── Dataset ───────────────────────────────────────────────────────────────────

class RealVLADataset(Dataset):
    """
    Sliding-window dataset over real humanoid episodes.

    Each sample
    -----------
    state_history : FloatTensor  (history_len, 69)
    future_state  : FloatTensor  (future_len,  69)
    image         : FloatTensor  (n_cams, 3, image_size, image_size)
    text          : str
    """

    _IMG_MEAN = [0.485, 0.456, 0.406]
    _IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root:   str,
        history_len: int  = 2,
        future_len:  int  = 32,
        image_size:  int  = 224,
        norm_stats:  dict = None,
        skip_images: bool = False,
    ):
        self.history_len = history_len
        self.future_len  = future_len
        self.window_len  = history_len + future_len
        self.norm_stats  = norm_stats
        self.skip_images = skip_images

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMG_MEAN, std=self._IMG_STD),
        ])

        self.windows = []
        episode_dirs = _sorted_episode_dirs(data_root)
        for ep_dir in episode_dirs:
            self._index_episode(ep_dir)

        print(
            f"[RealVLADataset] {len(self.windows)} windows  "
            f"from {len(episode_dirs)} episodes  "
            f"(history={history_len}, future={future_len})"
        )

    def _index_episode(self, ep_dir: str):
        features, envs, text = _load_episode(ep_dir)
        T = len(features)
        for env_name, img_paths in envs.items():
            for start in range(T - self.window_len + 1):
                self.windows.append({
                    'features':  features[start : start + self.window_len],
                    'img_paths': img_paths[start + self.history_len - 1],
                    'text':      text,
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w        = self.windows[idx]
        features = w['features']                               # (window_len, 67)

        history = features[: self.history_len].copy()         # (H, 67)
        future  = features[self.history_len :].copy()         # (F, 67)

        if self.norm_stats is not None:
            mu  = self.norm_stats['state_mean']
            sig = self.norm_stats['state_std']
            history = (history - mu) / sig
            future  = (future  - mu) / sig

        result = {
            'state_history': torch.from_numpy(history).float(),
            'future_state':  torch.from_numpy(future).float(),
            'text':          w['text'],
        }

        if not self.skip_images:
            result['image'] = torch.stack([
                self.transform(Image.open(p).convert('RGB'))
                for p in w['img_paths']
            ])   # (n_cams, 3, H, W)

        return result


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    result = {
        'state_history': torch.stack([b['state_history'] for b in batch]),
        'future_state':  torch.stack([b['future_state']  for b in batch]),
        'text':          [b['text'] for b in batch],
    }
    if 'image' in batch[0]:
        result['image'] = torch.stack([b['image'] for b in batch])
    return result


# ── Helper ────────────────────────────────────────────────────────────────────

def _sorted_episode_dirs(data_root: str):
    return sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if d.startswith('episode_') and os.path.isdir(os.path.join(data_root, d))
    ])


# ── Rollout Primitive Dataset ────────────────────────────────────────────────

class RolloutPrimitiveDataset(IterableDataset):
    """
    IterableDataset for rollout training (RobotMDAR-style slicing).

    Each iteration yields a list of ``num_primitive`` consecutive primitives,
    each already batched across B sequences.

    Primitive k covers features ``[seg_start + k*F : seg_start + k*F + H + F]``.
    Adjacent primitives overlap by H features::

        prim[k].future[-H:]  ==  prim[k+1].history

    Each element in the returned list is a dict::

        features  : FloatTensor (B, H+F, 69)  — normalised
        img_paths : list[list[str]]            — B × n_cams paths (history last frame)
        text      : list[str]                  — B text strings
    """

    _IMG_MEAN = [0.485, 0.456, 0.406]
    _IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root:     str,
        history_len:   int  = 2,
        future_len:    int  = 32,
        num_primitive: int  = 8,
        batch_size:    int  = 50,
        image_size:    int  = 224,
        norm_stats:    dict = None,
        cache_dir:     str  = None,
        max_episodes:  int  = None,
        episode_seed:  int  = 42,
        max_envs_per_episode: int = None,
        env_seed:      int  = 42,
    ):
        super().__init__()
        self.history_len   = history_len
        self.future_len    = future_len
        self.num_primitive = num_primitive
        self.batch_size    = batch_size
        self.norm_stats    = norm_stats
        self.segment_len   = history_len + future_len * num_primitive
        self.use_cache     = cache_dir is not None and os.path.isdir(cache_dir)
        self.max_envs_per_episode = max_envs_per_episode
        self.env_seed             = env_seed

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMG_MEAN, std=self._IMG_STD),
        ])

        # Load all (episode, env) pairs into memory
        # Each env within an episode shares the same features but has different images
        self.episodes = []
        episode_dirs  = _sorted_episode_dirs(data_root)
        n_total       = len(episode_dirs)
        if max_episodes is not None and max_episodes < n_total:
            rng           = np.random.RandomState(episode_seed)
            picked        = rng.choice(n_total, size=max_episodes, replace=False)
            episode_dirs  = sorted(episode_dirs[i] for i in picked)
            print(f"[RolloutPrimitiveDataset] sampled {max_episodes}/{n_total} "
                  f"episodes (seed={episode_seed})")
        skipped = 0
        cache_hits = 0
        missing_cache = []
        total_envs = 0
        for ep_dir in episode_dirs:
            features, envs, text = _load_episode(ep_dir)
            if len(features) < self.segment_len:
                skipped += 1
                continue
            if norm_stats is not None:
                mu  = norm_stats['state_mean']
                sig = norm_stats['state_std']
                features = ((features - mu) / sig).astype(np.float32)

            ep_name = os.path.basename(ep_dir)
            n_feats = len(features)

            env_items = list(envs.items())
            if (max_envs_per_episode is not None
                    and len(env_items) > max_envs_per_episode):
                seed = (hash(ep_name) ^ env_seed) & 0xFFFFFFFF
                rng  = np.random.RandomState(seed)
                picked = rng.choice(
                    len(env_items), size=max_envs_per_episode, replace=False)
                env_items = [env_items[i] for i in sorted(picked)]

            for env_name, img_paths in env_items:
                total_envs += 1
                ep_entry = {
                    'features':  features,
                    'img_paths': img_paths,
                    'text':      text,
                    'env_name':  env_name,
                    'ep_name':   ep_name,
                }

                # Load cached encoder features if available
                if self.use_cache:
                    cache_file = os.path.join(
                        cache_dir, f"{ep_name}__{env_name}.pt")
                    if os.path.isfile(cache_file):
                        cached = torch.load(
                            cache_file, map_location='cpu', weights_only=True)
                        n_cached = cached['vision_patches'].shape[0]
                        if n_cached < n_feats:
                            missing_cache.append(
                                f"  {ep_name}/{env_name}: cache has {n_cached} "
                                f"frames, features has {n_feats}")
                        else:
                            # vision_patches : (N, n_cams, N_patches, dinov2_dim) — raw DINOv2 patch tokens
                            # text_feat      : (clip_dim,)                        — raw CLIP pooler_output
                            ep_entry['vision_patches'] = cached['vision_patches'][:n_feats]
                            ep_entry['text_feat']      = cached['text_feat']
                            cache_hits += 1
                    else:
                        missing_cache.append(
                            f"  {ep_name}/{env_name}: {cache_file} not found")

                self.episodes.append(ep_entry)

        if self.use_cache:
            if missing_cache:
                raise RuntimeError(
                    f"[RolloutPrimitiveDataset] --cache_dir specified but "
                    f"{len(missing_cache)} envs have missing/invalid cache:\n"
                    + "\n".join(missing_cache)
                    + "\nRun cache_features.py to fix, or remove --cache_dir."
                )
            print(f"[RolloutPrimitiveDataset] Cache: {cache_hits}/{len(self.episodes)} "
                  f"envs — using cached encoder features "
                  f"(DINOv2 + CLIP skipped during training)")

        print(
            f"[RolloutPrimitiveDataset] {len(self.episodes)} samples "
            f"from {len(episode_dirs)} episodes × {total_envs} envs "
            f"(skipped {skipped}) | segment_len={self.segment_len}, "
            f"P={num_primitive}, H={history_len}, F={future_len}, B={batch_size}"
        )

    # ── Sampling ──────────────────────────────────────────────────────────

    def _sample_batch(self) -> list:
        """Sample one batch of P primitives from random episodes."""
        B = min(self.batch_size, len(self.episodes))
        indices = np.random.randint(0, len(self.episodes), size=B)

        primitives = [[] for _ in range(self.num_primitive)]

        for idx in indices:
            ep = self.episodes[idx]
            T  = len(ep['features'])
            max_start = T - self.segment_len
            seg_start = np.random.randint(0, max_start + 1)

            for pidx in range(self.num_primitive):
                prim_start = seg_start + pidx * self.future_len
                prim_end   = prim_start + self.history_len + self.future_len
                img_idx    = prim_start + self.history_len - 1

                prim_entry = {
                    'features':  ep['features'][prim_start:prim_end],
                    'img_paths': ep['img_paths'][img_idx],
                    'text':      ep['text'],
                }
                if self.use_cache:
                    # vision_patches: (N, n_cams, N_patches, dinov2_dim) → pick frame
                    prim_entry['vision_patches'] = ep['vision_patches'][img_idx]  # (n_cams, N_patches, dinov2_dim)
                    prim_entry['text_feat']      = ep['text_feat']                # (clip_dim,)

                primitives[pidx].append(prim_entry)

        result = []
        for pidx in range(self.num_primitive):
            img_paths_batch = [p['img_paths'] for p in primitives[pidx]]
            prim_dict = {
                'features':  torch.from_numpy(
                    np.stack([p['features'] for p in primitives[pidx]])).float(),
                'img_paths': img_paths_batch,
                'text':      [p['text'] for p in primitives[pidx]],
            }
            if self.use_cache:
                prim_dict['vision_patches'] = torch.stack(
                    [p['vision_patches'] for p in primitives[pidx]])  # (B, n_cams, N_patches, dinov2_dim)
                prim_dict['text_feat'] = torch.stack(
                    [p['text_feat'] for p in primitives[pidx]])       # (B, clip_dim)
            else:
                # Load + transform images inside the dataset so DataLoader
                # workers can parallelise the disk IO.
                prim_dict['images'] = torch.stack([
                    torch.stack([
                        self.transform(Image.open(p).convert('RGB'))
                        for p in paths
                    ])
                    for paths in img_paths_batch
                ])                                                # (B, n_cams, 3, H, W)
            result.append(prim_dict)
        return result

    def __iter__(self):
        while True:
            yield self._sample_batch()


# ── Image loader helper ──────────────────────────────────────────────────────

def load_images_batch(
    img_paths_batch: list,
    transform: transforms.Compose,
) -> torch.Tensor:
    """
    Load and transform images for a batch of samples.

    Parameters
    ----------
    img_paths_batch : list of B lists, each with n_cams file paths
    transform       : torchvision transform pipeline

    Returns
    -------
    torch.Tensor of shape (B, n_cams, 3, H, W)
    """
    batch = []
    for paths in img_paths_batch:
        imgs = [transform(Image.open(p).convert('RGB')) for p in paths]
        batch.append(torch.stack(imgs))
    return torch.stack(batch)
