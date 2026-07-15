"""
VLA Backbone: Vision-Language-Action Policy for Humanoid Control
=================================================================
Direct action-space diffusion via Rectified Flow Matching,
with a DP-Transformer (cross-attention) denoiser.

  Conditioning
  ────────────
    text  (CLIP pooler)       → 1 token
    image (DINOv2 patches →    48 tokens  (3 cams × 4×4 avg-pool, + cam/pos emb)
           4×4 avg pool)
    state_history (H frames)  → H tokens
    flow time t ∈ [0,1]       → 1 token
    └─ all concatenated as K/V stream into cross-attention

  Training (Rectified Flow)
  ─────────────────────────
    x_0 = action_gt (already normalised by caller);   x_1 ~ N(0, I)
    t   ~ Uniform(0, 1)
    x_t = (1 - t)·x_0 + t·x_1
    v*  = x_1 - x_0
    v̂   = DenoiserDPT(x_t, t, text, vision, state)
    Loss: MSE(v̂, v*)

  Inference (Euler)
  ─────────────────
    x ← N(0, I)
    for k in range(N):
        t_cur = 1 - k/N
        x    -= (1/N) · DenoiserDPT(x, t_cur, cond)
    return x                       # in normalised space; caller denormalises

  Normalisation contract
  ──────────────────────
    The backbone operates entirely in the normalised 69D space.
    - Training:   data.py normalises features before they reach the model.
    - Deployment: deploy scripts normalise inputs and denormalise outputs
                  using the paired ``norm_stats.npz``.
    The model itself never reads or applies normalisation stats.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


# ---------------------------------------------------------------------------
# 1. Sinusoidal time embedding for continuous t ∈ [0, 1]
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Continuous flow-matching time → (B, dim) sinusoidal embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) values in [0, 1]
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs             # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# 2. State Encoder  (per-frame state → model_dim)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Works on any trailing state_dim; Linear applies per-token.
        return self.net(state)


# ---------------------------------------------------------------------------
# 3. Flow Matching Scheduler  (Rectified Flow)
# ---------------------------------------------------------------------------

class FlowMatchingScheduler:
    """
    Rectified Flow scheduler.

    Convention: x_0 = data, x_1 = noise.
        x_t    = (1 - t) · x_0 + t · x_1,          t ∈ [0, 1]
        v(x_t) = dx_t/dt = x_1 - x_0
    """

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        x0 : (B, T, D)
        t  : (B,)
        returns: x_t (same shape as x0), v_true (same shape)
        """
        noise = torch.randn_like(x0)
        t_shape = [x0.shape[0]] + [1] * (x0.dim() - 1)
        t_exp   = t.view(t_shape)
        x_t     = (1.0 - t_exp) * x0 + t_exp * noise
        v_true  = noise - x0
        return x_t, v_true

    @torch.no_grad()
    def euler_sample(
        self,
        denoiser:   nn.Module,
        shape:      tuple,
        text_emb:   torch.Tensor,
        vision_emb: torch.Tensor,
        state_emb:  torch.Tensor,
        num_steps:  int = 10,
    ) -> torch.Tensor:
        device = text_emb.device
        x  = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_cur   = 1.0 - k * dt
            t_batch = torch.full((shape[0],), t_cur, device=device)
            v       = denoiser(x, t_batch, text_emb, vision_emb, state_emb)
            x       = x - dt * v
        return x   # ≈ x_0


# ---------------------------------------------------------------------------
# 4. Action Denoiser — DP-Transformer with cross-attention
# ---------------------------------------------------------------------------

class ActionDenoiserDPT(nn.Module):
    """
    DP-Transformer style denoiser operating directly on the action sequence.

    Q stream: action tokens (future_len) with self-attention between them.
    K/V stream: [t_token | text_token | vision_tokens | state_tokens],
                injected via cross-attention in every decoder block.
    """

    def __init__(
        self,
        action_dim:  int,
        future_len:  int,
        h_dim:       int   = 256,
        num_layers:  int   = 6,
        num_heads:   int   = 8,
        ff_size:     int   = 1024,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.future_len = future_len
        self.h_dim      = h_dim

        # Action stream
        self.action_embed   = nn.Linear(action_dim, h_dim)
        self.action_pos_emb = nn.Parameter(torch.randn(future_len, h_dim) * 0.02)

        # Flow-matching time embedding (continuous t ∈ [0, 1])
        self.time_embed = SinusoidalTimeEmbedding(h_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim),
        )

        # Per-sub-stream type embedding so cross-attention can distinguish
        # time / text / vision / state tokens (index order: 0=time, 1=text, 2=vision, 3=state).
        # Per-modality *layout* embeddings (vision cam/pos, state temporal pos)
        # live in VLABackbone at h_dim and are baked into vision_emb / state_emb
        # before they reach the denoiser.
        self.kv_type_emb = nn.Parameter(torch.randn(4, h_dim) * 0.02)

        # Transformer decoder (self-attn + cross-attn + FFN) with pre-norm for stability
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=h_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output: per-action-step velocity prediction
        self.output_proj = nn.Linear(h_dim, action_dim)

    def forward(
        self,
        x_t:        torch.Tensor,   # (B, T_action, action_dim)
        t:          torch.Tensor,   # (B,)  in [0, 1]
        text_emb:   torch.Tensor,   # (B, h_dim)
        vision_emb: torch.Tensor,   # (B, N_v, h_dim)
        state_emb:  torch.Tensor,   # (B, H,   h_dim)
    ) -> torch.Tensor:              # (B, T_action, action_dim)

        # Action tokens (Q)
        a = self.action_embed(x_t) + self.action_pos_emb.unsqueeze(0)   # (B, T, h)

        # KV stream — each sub-stream adds its type embedding so the decoder
        # can tell "this is the time token / text / vision / state" apart.
        # Condition tokens already live in h_dim (projected by VLABackbone).
        t_tok     = self.time_mlp(self.time_embed(t)).unsqueeze(1) + self.kv_type_emb[0]   # (B, 1,   h)
        text_kv   = text_emb.unsqueeze(1)                          + self.kv_type_emb[1]   # (B, 1,   h)
        vision_kv = vision_emb                                     + self.kv_type_emb[2]   # (B, N_v, h)
        state_kv  = state_emb                                      + self.kv_type_emb[3]   # (B, H,   h)
        cond_kv   = torch.cat([t_tok, text_kv, vision_kv, state_kv], dim=1)                # (B, 2+N_v+H, h)

        # Cross-attention decoder
        out = self.decoder(tgt=a, memory=cond_kv)                      # (B, T, h)
        return self.output_proj(out)                                   # (B, T, action_dim)


# ---------------------------------------------------------------------------
# 5. VLA Backbone
# ---------------------------------------------------------------------------

class VLABackbone(nn.Module):
    """
    VLA backbone with:
      - DINOv2 (frozen) → patch tokens → 4×4 avg pool → 48 vision tokens
      - CLIP (frozen)   → text pooler  → 1 text token
      - State history   → H=10 tokens
      - Flow time t     → 1 token
      - Action denoiser : DP-Transformer (cross-attention) operating on a
                          future_len-step action sequence
      - Training        : Rectified Flow Matching
      - Inference       : Euler, N_steps (default 10)

    Training
    --------
        loss, info = model.flow_matching_loss(text, image, state_history, action_gt)

    Inference
    ---------
        actions = model.sample(text, image, state_history)     # (B, future_len, action_dim)
    """

    def __init__(
        self,
        action_dim:          int   = 69,
        history_len:         int   = 2,
        future_len:          int   = 32,
        n_cams:              int   = 3,
        # Single shared model dim: encoder projections + denoiser internal width.
        model_dim:           int   = 512,
        # Denoiser
        denoiser_layers:     int   = 8,
        denoiser_heads:      int   = 8,
        denoiser_ff_size:    int   = 2048,
        dropout:             float = 0.1,
        # Flow matching
        num_inference_steps: int   = 10,
        # Encoders. Paths are deliberately local-only; injected modules make
        # architecture tests independent of large pretrained assets.
        clip_model_name:     str | None = None,
        dinov2_model_name:   str | None = None,
        dinov2_variant:      str   = "dinov2_vitb14",
        freeze_encoders:     bool  = True,
        clip_tokenizer=None,
        clip_text_model: nn.Module | None = None,
        dinov2: nn.Module | None = None,
    ):
        super().__init__()
        self.action_dim          = action_dim
        self.history_len         = history_len
        self.future_len          = future_len
        self.n_cams              = n_cams
        self.num_inference_steps = num_inference_steps

        # ── Multimodal frontend ──────────────────────────────────────────────
        if (clip_tokenizer is None) != (clip_text_model is None):
            raise ValueError(
                "clip_tokenizer and clip_text_model must be injected together"
            )
        if clip_tokenizer is None:
            if clip_model_name is None:
                raise ValueError(
                    "clip_model_name must point to a local CLIP snapshot when "
                    "CLIP modules are not injected"
                )
            clip_path = Path(clip_model_name).expanduser()
            if not clip_path.is_dir():
                raise FileNotFoundError(f"local CLIP model not found: {clip_path}")
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                str(clip_path), local_files_only=True
            )
            clip_text_model = CLIPTextModel.from_pretrained(
                str(clip_path), local_files_only=True
            )
        self.clip_tokenizer = clip_tokenizer
        self.clip_text_model = clip_text_model
        clip_text_dim        = self.clip_text_model.config.hidden_size

        if dinov2 is None:
            if dinov2_model_name is None:
                raise ValueError(
                    "dinov2_model_name must point to a local torch.hub repo when "
                    "DINOv2 is not injected"
                )
            dinov2_path = Path(dinov2_model_name).expanduser()
            if not dinov2_path.is_dir():
                raise FileNotFoundError(f"local DINOv2 repo not found: {dinov2_path}")
            dinov2 = torch.hub.load(
                str(dinov2_path), dinov2_variant, pretrained=True, source='local'
            )
        self.dinov2 = dinov2
        dinov2_dim   = self.dinov2.embed_dim

        self._encoders_frozen = freeze_encoders
        if freeze_encoders:
            self._freeze(self.clip_text_model)
            self._freeze(self.dinov2)

        self.text_proj     = nn.Linear(clip_text_dim, model_dim)
        self.vision_proj   = nn.Linear(dinov2_dim,    model_dim)   # per-token projection
        self.state_encoder = StateEncoder(action_dim, model_dim)

        # Vision patch pool: 16×16 grid → 8×8 via avg pool → 64 tokens / cam.
        self.vision_pool_side = 8
        n_pool_tokens         = self.vision_pool_side * self.vision_pool_side
        self.cam_emb          = nn.Parameter(torch.randn(n_cams,        model_dim) * 0.02)
        self.vision_pos_emb   = nn.Parameter(torch.randn(n_pool_tokens, model_dim) * 0.02)
        self.state_pos_emb    = nn.Parameter(torch.randn(history_len,   model_dim) * 0.02)

        # ── Denoiser ─────────────────────────────────────────────────────────
        self.denoiser = ActionDenoiserDPT(
            action_dim=action_dim, future_len=future_len,
            h_dim=model_dim, num_layers=denoiser_layers, num_heads=denoiser_heads,
            ff_size=denoiser_ff_size, dropout=dropout,
        )

        # ── Flow matching scheduler ──────────────────────────────────────────
        self.scheduler = FlowMatchingScheduler()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False
        module.eval()

    def train(self, mode: bool = True):
        # Keep frozen pretrained encoders in eval mode so Dropout / BN running
        # stats don't drift during fine-tuning of the rest of the model.
        super().train(mode)
        if getattr(self, '_encoders_frozen', False):
            self.clip_text_model.eval()
            self.dinov2.eval()
        return self

    def _pool_and_project_patches(self, patch_flat: torch.Tensor, B: int) -> torch.Tensor:
        """
        DINOv2 patch tokens → grid → avg-pool → per-token projection → + cam_emb + pos_emb.

        patch_flat : (B * n_cams, N_patches, dinov2_dim)   typically (B*n_cams, 256, 768)
        returns    : (B, n_cams * pool_side², model_dim)   typically (B, 48, model_dim)
        """
        BN, N_patches, D_vis = patch_flat.shape
        grid_side            = int(N_patches ** 0.5)            # 16 for 224/14
        assert grid_side * grid_side == N_patches, (
            f"expected square patch grid, got N_patches={N_patches} "
            f"(grid_side={grid_side}); check DINOv2 version / input resolution"
        )
        assert grid_side % self.vision_pool_side == 0, (
            f"grid_side={grid_side} not divisible by vision_pool_side="
            f"{self.vision_pool_side}"
        )
        pool_k = grid_side // self.vision_pool_side

        grid   = patch_flat.reshape(BN, grid_side, grid_side, D_vis).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(grid, kernel_size=pool_k, stride=pool_k)
        pooled = pooled.flatten(2).transpose(1, 2)                        # (BN, pool², D_vis)

        pooled = self.vision_proj(pooled)                                 # (BN, pool², model_dim)

        n_pool = pooled.shape[1]
        pooled = pooled.reshape(B, self.n_cams, n_pool, -1)

        pooled = pooled \
            + self.cam_emb[None, :, None, :] \
            + self.vision_pos_emb[None, None, :, :]

        return pooled.reshape(B, self.n_cams * n_pool, -1)                # (B, 48, model_dim)

    # ── Condition encoding ───────────────────────────────────────────────────

    @torch.no_grad()
    def encode_frozen(self, text, image):
        """
        Run only the frozen pretrained encoders (CLIP text + DINOv2 vision) and
        return their raw outputs. Safe to cache across multiple forward calls
        within a training step — outputs don't depend on trainable parameters.

        Returns:
            text_feat      : (B, clip_dim)                         — CLIP pooler_output
            vision_patches : (B, n_cams, N_patches, dinov2_dim)    — DINOv2 patch tokens
        """
        device = self.text_proj.weight.device

        tokens = self.clip_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=77,
        ).to(device)
        text_feat = self.clip_text_model(**tokens).pooler_output             # (B, clip_dim)

        B          = image.shape[0]
        imgs_flat  = image.reshape(B * self.n_cams, *image.shape[2:])
        patch_flat = self.dinov2.forward_features(imgs_flat)['x_norm_patchtokens']
        #           (B*n_cams, N, D_vis)
        vision_patches = patch_flat.reshape(B, self.n_cams, *patch_flat.shape[1:])
        #           (B, n_cams, N, D_vis)

        return text_feat, vision_patches

    def _encode_condition(self, text, image, state_history):
        """
        Encode multimodal conditioning signals.

        Returns:
            text_emb   : (B, model_dim)             — single CLIP pooler token
            vision_emb : (B, 48,          model_dim) — 48 pooled DINOv2 patch tokens
            state_emb  : (B, history_len, model_dim) — per-frame state tokens
        """
        text_feat, vision_patches = self.encode_frozen(text, image)
        return self._encode_condition_cached(text_feat, vision_patches, state_history)

    def _encode_condition_cached(
        self,
        text_emb_raw:   torch.Tensor,   # (B, clip_dim) — raw CLIP pooler_output
        vision_patches: torch.Tensor,   # (B, n_cams, N_patches, dinov2_dim) — raw DINOv2 patch tokens
        state_history:  torch.Tensor,   # (B, H, state_dim)
    ):
        """
        Encode conditioning from pre-computed encoder features.
        Skips CLIP and DINOv2 forward — only runs pool + projections.
        """
        device         = self.text_proj.weight.device
        text_emb_raw   = text_emb_raw.to(device)
        vision_patches = vision_patches.to(device)

        text_emb = self.text_proj(text_emb_raw)                          # (B, D)

        B, n_cams, N_patches, D_vis = vision_patches.shape
        patch_flat = vision_patches.reshape(B * n_cams, N_patches, D_vis)
        vision_emb = self._pool_and_project_patches(patch_flat, B)       # (B, 48, D)

        state_emb = self.state_encoder(state_history)                    # (B, H, D)
        state_emb = state_emb + self.state_pos_emb.unsqueeze(0)          # + temporal pos emb

        return text_emb, vision_emb, state_emb

    # ── Training: Flow Matching loss ─────────────────────────────────────────

    def flow_matching_loss(
        self,
        text:          list,
        image:         torch.Tensor,   # (B, n_cams, 3, H, W)
        state_history: torch.Tensor,   # (B, H, action_dim) — already normalised by the data pipeline
        action_gt:     torch.Tensor,   # (B, future_len, action_dim) — already normalised
    ) -> tuple:
        """Rectified-flow MSE in normalised action space. Caller owns normalisation."""
        text_emb, vision_emb, state_emb = self._encode_condition(text, image, state_history)

        B = action_gt.shape[0]
        t = torch.rand(B, device=action_gt.device)

        x_t, v_true = self.scheduler.q_sample(action_gt, t)
        v_pred      = self.denoiser(x_t, t, text_emb, vision_emb, state_emb)

        loss = F.mse_loss(v_pred, v_true)
        return loss, {'flow_mse': loss.item()}

    def flow_matching_loss_cached(
        self,
        text_emb_raw:   torch.Tensor,   # (B, clip_dim)
        vision_patches: torch.Tensor,   # (B, n_cams, N_patches, dinov2_dim)
        state_history:  torch.Tensor,   # (B, H, action_dim) — already normalised
        action_gt:      torch.Tensor,   # (B, future_len, action_dim) — already normalised
    ) -> tuple:
        
        # proj encoder outputs的维度
        text_emb, vision_emb, state_emb = self._encode_condition_cached(
            text_emb_raw, vision_patches, state_history)

        B = action_gt.shape[0]
        t = torch.rand(B, device=action_gt.device)

        x_t, v_true = self.scheduler.q_sample(action_gt, t)
        v_pred      = self.denoiser(x_t, t, text_emb, vision_emb, state_emb)

        loss = F.mse_loss(v_pred, v_true)
        return loss, {'flow_mse': loss.item()}

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        text:          list,
        image:         torch.Tensor,   # (B, n_cams, 3, H, W)
        state_history: torch.Tensor,   # (B, H, action_dim) — already normalised
    ) -> torch.Tensor:
        """
        Euler integration of the learned velocity field from noise (t=1) to
        data (t=0). Returns action sequence in normalised space
        (B, future_len, action_dim). Caller is responsible for denormalisation.
        """
        text_emb, vision_emb, state_emb = self._encode_condition(text, image, state_history)

        B     = state_history.shape[0]
        shape = (B, self.future_len, self.action_dim)

        return self.scheduler.euler_sample(
            self.denoiser, shape, text_emb, vision_emb, state_emb,
            num_steps=self.num_inference_steps,
        )

    @torch.no_grad()
    def sample_cached(
        self,
        text_emb_raw:   torch.Tensor,
        vision_patches: torch.Tensor,
        state_history:  torch.Tensor,
    ) -> torch.Tensor:
        text_emb, vision_emb, state_emb = self._encode_condition_cached(
            text_emb_raw, vision_patches, state_history)

        B     = state_history.shape[0]
        shape = (B, self.future_len, self.action_dim)

        return self.scheduler.euler_sample(
            self.denoiser, shape, text_emb, vision_emb, state_emb,
            num_steps=self.num_inference_steps,
        )

    def forward(
        self,
        text:          list,
        image:         torch.Tensor,
        state_history: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample(text, image, state_history)


# ---------------------------------------------------------------------------
# Smoke-test  (python vla_backbone.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B          = 2
    ACTION_DIM = 69
    HIST_LEN   = 10
    FUTURE_LEN = 32
    N_CAMS     = 3

    print("Building VLABackbone …")
    model = VLABackbone(
        action_dim=ACTION_DIM, history_len=HIST_LEN, future_len=FUTURE_LEN, n_cams=N_CAMS,
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params — trainable: {n_train:,}  total: {n_total:,}")

    image         = torch.randn(B, N_CAMS, 3, 224, 224).to(device)
    state_history = torch.randn(B, HIST_LEN, ACTION_DIM).to(device)
    action_gt     = torch.randn(B, FUTURE_LEN, ACTION_DIM).to(device)
    text          = ["pick up the cube", "move arm left"]

    print("Flow-matching loss …")
    model.train()
    loss, info = model.flow_matching_loss(text, image, state_history, action_gt)
    print(f"  loss={loss.item():.4f}  flow_mse={info['flow_mse']:.4f}")

    print("Inference (Euler 10 steps) …")
    model.eval()
    actions = model.sample(text, image, state_history)
    print(f"  actions shape: {actions.shape}  (expect ({B}, {FUTURE_LEN}, {ACTION_DIM}))")
    assert actions.shape == (B, FUTURE_LEN, ACTION_DIM)
    print("Smoke-test passed.")