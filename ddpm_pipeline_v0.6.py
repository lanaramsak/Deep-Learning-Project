"""
DDPM Face Generation — Training Dynamics Overhaul (v0.6)
=========================================================
Goal: Same architecture as v0.5 — dramatically better training dynamics.
      Fix the Cosine-collapse (v0.5 Phase 3) and DPM-Solver failure (v0.5 Phase 4)
      observed in the previous run.

Four documented improvement phases (each builds on the previous):
  Phase 1 – Baseline U-Net (4 stages), no attention, linear schedule
  Phase 2 – + Self-attention at 16×16 AND 32×32
  Phase 3 – + Shifted-cosine noise schedule
  Phase 4 – + DPM-Solver++(2M) sampler (50 steps — robust)

Changes vs v0.5:
  [LOSS]  v-prediction instead of ε-prediction (Salimans & Ho 2022)
          Empirically much better for small/medium models; pairs especially
          well with cosine schedules. Loss target becomes v = α·ε − σ·x₀.

  [LOSS]  Min-SNR loss weighting (Hang et al. 2024)
          Weights the per-sample loss by min(SNR(t), γ) with γ=5, focusing
          training on informative timesteps. Produces ~10% FID reduction at
          no compute cost. Loss: λ(t)·||pred − target||²

  [SCHED] Shifted cosine schedule for 128×128 (Hoogeboom et al. 2023)
          Standard cosine was calibrated for 32×32/64×64. At 128×128 it is
          too aggressive — noise overpowers signal too early and the model
          never learns high-frequency detail. Shift factor = 2 (image_size/64)
          makes the schedule behave like linear-at-64×64 when scaled up.

  [SAMP]  Phase 4 uses DPM-Solver++ with 50 steps (was 20)
          20 steps completely failed at this model size (FID 444 in v0.5).
          50 steps gives ~20× speedup over DDPM 1000 while remaining robust.

  [TRAIN] EMA warmup: EMA starts after 2000 training steps (was: from step 1)
          Initial random weights shouldn't be averaged into the shadow model.

Kept from v0.5:
  - 128×128 resolution, 4-stage U-Net [64, 128, 256, 512]
  - Multi-scale attention (16×16 + 32×32) in Phase 2+
  - batch_size=32 × grad_accum=2 = effective 64
  - LR warmup 1000 steps → cosine annealing
  - AdamW, dropout=0.05, FID reference deterministic via MASTER_SEED

Usage on HPC:
  sbatch -p normal --qos gpu_batch --gres=gpu:1 --time=16:00:00 \
    --job-name "v06_p2" \
    --wrap "python3 ~/ddpm_pipeline_v0.6.py --phase 2 --epochs 80 --log_dir /data/01/up202512956/ddpm_runs_v06"
"""

import os, sys, math, copy, json, time, argparse, logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as ckpt_fn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[warn] scipy not found — FID will be skipped.")

# ──────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────

BASE_DIR = Path(os.environ.get("DDPM_BASE_DIR", "/data/01/up202402612/data"))
REAL_DIR = BASE_DIR / "wiki"

MASTER_SEED = 42


@dataclass
class Config:
    # ── data ──────────────────────────────────────────────────
    data_dirs:    List[str] = field(default_factory=lambda: [str(REAL_DIR)])
    image_size:   int       = 128
    val_split:    float     = 0.10
    num_workers:  int       = 4

    # ── model ─────────────────────────────────────────────────
    channels:     List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_attention: bool     = False
    attn_resolutions: List[int] = field(default_factory=lambda: [16, 32])
    dropout:      float     = 0.05
    use_checkpoint: bool    = False

    # ── noise schedule ────────────────────────────────────────
    T:            int       = 1000
    # [v0.6] "linear" | "cosine" | "shifted_cosine"
    schedule:     str       = "linear"
    beta_start:   float     = 1e-4
    beta_end:     float     = 2e-2
    # [v0.6] Shift factor for shifted-cosine schedule (Hoogeboom et al. 2023).
    # For 128×128, shift=2.0 (= 128/64) effectively rescales the schedule so
    # signal/noise ratio matches that of 64×64 linear at each timestep.
    cosine_shift: float     = 2.0

    # ── loss ──────────────────────────────────────────────────
    # [v0.6] "eps" (v0.5 behavior) | "v" (Salimans & Ho 2022)
    prediction_type: str    = "v"
    # [v0.6] "uniform" (v0.5 behavior) | "min_snr" (Hang et al. 2024)
    loss_weighting: str     = "min_snr"
    min_snr_gamma:  float   = 5.0  # Clipping constant for Min-SNR

    # ── training ──────────────────────────────────────────────
    epochs:                 int   = 80
    batch_size:             int   = 32
    grad_accum_steps:       int   = 2
    lr:                     float = 2e-4
    lr_min:                 float = 1e-6
    warmup_steps:           int   = 1000
    weight_decay:           float = 1e-4
    ema_decay:              float = 0.9995
    ema_warmup_steps:       int   = 2000  # [v0.6] EMA starts after this many updates
    use_fp16:               bool  = True
    grad_clip:              float = 1.0

    # ── sampling ──────────────────────────────────────────────
    sampler:      str  = "ddpm"
    sample_steps: int  = 1000
    n_samples:    int  = 16

    # ── logging ───────────────────────────────────────────────
    log_dir:          str  = "./ddpm_runs_v06"
    sample_every:     int  = 10
    fid_every:        int  = 20
    checkpoint_every: int  = 20
    phase_name:       str  = "phase1_baseline"
    fid_n_samples:    int  = 2048
    final_gallery_n:  int  = 100


def get_phase_config(phase: int, **overrides) -> Config:
    cfg = Config()
    if phase >= 2:
        cfg.use_attention = True
        cfg.phase_name = "phase2_attention"
    if phase >= 3:
        # [v0.6] Use shifted cosine instead of standard cosine at 128×128
        cfg.schedule = "shifted_cosine"
        cfg.phase_name = "phase3_shifted_cosine"
    if phase >= 4:
        cfg.sampler = "dpm_solver"
        # [v0.6] 50 steps is the sweet spot — 20 failed catastrophically in v0.5
        cfg.sample_steps = 50
        cfg.phase_name = "phase4_dpm_solver_50"
    if phase == 1:
        cfg.phase_name = "phase1_baseline"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ──────────────────────────────────────────────────────────────
# 2.  DATASET  (unchanged from v0.5)
# ──────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, data_dirs: List[str], image_size: int = 128, augment: bool = True):
        self.image_size = image_size
        self.paths: List[Path] = []
        for d in data_dirs:
            p = Path(d)
            if not p.exists():
                print(f"[warn] directory not found: {p}")
                continue
            self.paths.extend(p.rglob("*.jpg"))
            self.paths.extend(p.rglob("*.png"))
        self.paths.sort()
        print(f"Dataset: {len(self.paths):,} images from {len(data_dirs)} dir(s)")
        self.transform = self._build_transform(augment)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        tf_list: list = []
        if augment:
            tf_list.append(transforms.RandomHorizontalFlip())
        tf_list += [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        return transforms.Compose(tf_list)

    def without_augmentation(self) -> "FaceDataset":
        view = FaceDataset.__new__(FaceDataset)
        view.image_size = self.image_size
        view.paths      = self.paths
        view.transform  = self._build_transform(augment=False)
        return view

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_dataloaders(cfg: Config):
    full = FaceDataset(cfg.data_dirs, cfg.image_size, augment=True)
    n_val   = int(len(full) * cfg.val_split)
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(MASTER_SEED)
    )
    val_set.dataset = full.without_augmentation()

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader, val_set


# ──────────────────────────────────────────────────────────────
# 3.  NOISE SCHEDULER  [v0.6: + shifted cosine + v-prediction helpers]
# ──────────────────────────────────────────────────────────────

class NoiseScheduler:
    """
    [v0.6] Now supports three schedules:
      - linear
      - cosine (standard Nichol & Dhariwal 2021)
      - shifted_cosine (Hoogeboom et al. 2023, for high resolutions)

    Also exposes SNR (signal-to-noise ratio) for Min-SNR loss weighting and
    conversion helpers for v-prediction (Salimans & Ho 2022).
    """

    def __init__(self, T=1000, schedule="linear", beta_start=1e-4, beta_end=2e-2,
                 cosine_shift=2.0):
        self.T = T
        self.schedule_name = schedule

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        elif schedule == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod_full = f / f[0]
            alphas_cumprod = alphas_cumprod_full[1:].float()
            betas = (1.0 - alphas_cumprod_full[1:] / alphas_cumprod_full[:-1]).float().clamp(1e-4, 0.9999)
        elif schedule == "shifted_cosine":
            # [v0.6] Hoogeboom et al. 2023: shifted cosine for high resolutions.
            # Construct standard cosine ᾱ(t), then shift:
            #   logSNR_shifted(t) = logSNR(t) - 2*log(shift)
            # Equivalent operation on ᾱ:
            #   ᾱ_shifted = ᾱ / (ᾱ + shift² · (1 - ᾱ))
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod_full = f / f[0]
            shift_sq = cosine_shift ** 2
            alphas_cumprod_full = alphas_cumprod_full / (
                alphas_cumprod_full + shift_sq * (1 - alphas_cumprod_full)
            )
            alphas_cumprod = alphas_cumprod_full[1:].float()
            betas = (1.0 - alphas_cumprod_full[1:] / alphas_cumprod_full[:-1]).float().clamp(1e-4, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas              = betas
        self.alphas             = alphas
        self.alphas_cumprod     = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod           = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # SNR(t) = ᾱ(t) / (1 - ᾱ(t))  — used for Min-SNR loss weighting
        self.snr = alphas_cumprod / (1 - alphas_cumprod).clamp(min=1e-20)

        self.posterior_variance  = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_log_var   = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_c1   = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        self.posterior_mean_c2   = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)

    def to(self, device):
        for attr in vars(self):
            v = getattr(self, attr)
            if isinstance(v, torch.Tensor):
                setattr(self, attr, v.to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s * x0 + s1 * noise, noise

    def get_v(self, x0, noise, t):
        """[v0.6] Compute v-prediction target: v = α·ε − σ·x₀."""
        s  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s * noise - s1 * x0

    def v_to_eps(self, v, xt, t):
        """[v0.6] Convert v-prediction back to ε-prediction (needed for sampling)."""
        s  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s * v + s1 * xt

    def predict_x0(self, xt, t, eps):
        s  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (xt - s1 * eps) / s

    def p_mean_var(self, eps, xt, t):
        x0_pred = self.predict_x0(xt, t, eps).clamp(-1, 1)
        mean = (
            self.posterior_mean_c1[t][:, None, None, None] * x0_pred
            + self.posterior_mean_c2[t][:, None, None, None] * xt
        )
        log_var = self.posterior_log_var[t][:, None, None, None]
        return mean, log_var


# ──────────────────────────────────────────────────────────────
# 4.  LOSS WEIGHTING  [v0.6 NEW]
# ──────────────────────────────────────────────────────────────

def compute_loss_weights(snr: torch.Tensor, t: torch.Tensor,
                         weighting: str, gamma: float,
                         prediction_type: str) -> torch.Tensor:
    """
    Compute per-sample loss weights.

    Min-SNR-γ weighting (Hang et al. 2024):
      For ε-prediction: w(t) = min(SNR(t), γ) / SNR(t)
      For v-prediction: w(t) = min(SNR(t), γ) / (SNR(t) + 1)

    The v-prediction case includes +1 because v-prediction implicitly
    normalises: Var[v] = α² + σ² = 1, whereas Var[ε] = 1 and Var[x₀] varies.
    """
    snr_t = snr[t]
    if weighting == "uniform":
        return torch.ones_like(snr_t)
    elif weighting == "min_snr":
        min_snr = snr_t.clamp(max=gamma)
        if prediction_type == "v":
            return min_snr / (snr_t + 1)
        else:  # eps-prediction
            return min_snr / snr_t
    else:
        raise ValueError(f"Unknown loss_weighting: {weighting!r}")


# ──────────────────────────────────────────────────────────────
# 5.  U-NET  (unchanged from v0.5 — same architecture)
# ──────────────────────────────────────────────────────────────

def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.05):
        super().__init__()
        self.norm1  = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(min(32, out_ch), out_ch)
        self.drop   = nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, n_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.05, use_attention=False, n_heads=8):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch, n_heads) if use_attention else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res2(self.res1(x, t), t)
        x = self.attn(x)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim, dropout=0.05, use_attention=False, n_heads=8):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, t_dim, dropout)
        self.attn = SelfAttention(out_ch, n_heads) if use_attention else nn.Identity()

    def forward(self, x, skip, t):
        x = torch.cat([self.up(x), skip], dim=1)
        x = self.res2(self.res1(x, t), t)
        return self.attn(x)


class UNet(nn.Module):
    """4-stage U-Net for 128×128. Identical to v0.5."""

    def __init__(self, in_ch=3, channels=None, use_attention=True,
                 attn_resolutions=None, image_size=128, dropout=0.05,
                 use_checkpoint=False):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]
        assert len(channels) == 4
        if attn_resolutions is None:
            attn_resolutions = [16, 32]
        C0, C1, C2, C3 = channels
        t_dim = C0 * 4
        self.use_checkpoint = use_checkpoint

        self.time_mlp = nn.Sequential(
            nn.Linear(C0, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, C0, 3, padding=1)

        res_d1 = image_size
        res_d2 = image_size // 2
        res_d3 = image_size // 4
        res_mid = image_size // 8

        attn_d1 = use_attention and (res_d1 in attn_resolutions)
        attn_d2 = use_attention and (res_d2 in attn_resolutions)
        attn_d3 = use_attention and (res_d3 in attn_resolutions)
        attn_mid = use_attention and (res_mid in attn_resolutions)

        self.down1 = DownBlock(C0, C1, t_dim, dropout, attn_d1)
        self.down2 = DownBlock(C1, C2, t_dim, dropout, attn_d2)
        self.down3 = DownBlock(C2, C3, t_dim, dropout, attn_d3)

        self.mid1 = ResBlock(C3, C3, t_dim, dropout=0.0)
        self.mid_attn = SelfAttention(C3, n_heads=8) if attn_mid else nn.Identity()
        self.mid2 = ResBlock(C3, C3, t_dim, dropout=0.0)

        self.up1 = UpBlock(C3, C3, C2, t_dim, dropout, attn_d3)
        self.up2 = UpBlock(C2, C2, C1, t_dim, dropout, attn_d2)
        self.up3 = UpBlock(C1, C1, C0, t_dim, dropout, attn_d1)

        self.out_norm = nn.GroupNorm(min(32, C0), C0)
        self.out_conv = nn.Conv2d(C0, in_ch, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_emb(t, self.time_mlp[0].in_features))
        x = self.in_conv(x)

        if self.use_checkpoint and self.training:
            x, skip1 = ckpt_fn(self.down1, x, t_emb, use_reentrant=False)
            x, skip2 = ckpt_fn(self.down2, x, t_emb, use_reentrant=False)
            x, skip3 = ckpt_fn(self.down3, x, t_emb, use_reentrant=False)
            x = ckpt_fn(self.mid1, x, t_emb, use_reentrant=False)
            x = ckpt_fn(self.mid_attn, x, use_reentrant=False) if not isinstance(self.mid_attn, nn.Identity) else self.mid_attn(x)
            x = ckpt_fn(self.mid2, x, t_emb, use_reentrant=False)
            x = ckpt_fn(self.up1, x, skip3, t_emb, use_reentrant=False)
            x = ckpt_fn(self.up2, x, skip2, t_emb, use_reentrant=False)
            x = ckpt_fn(self.up3, x, skip1, t_emb, use_reentrant=False)
        else:
            x, skip1 = self.down1(x, t_emb)
            x, skip2 = self.down2(x, t_emb)
            x, skip3 = self.down3(x, t_emb)
            x = self.mid1(x, t_emb)
            x = self.mid_attn(x)
            x = self.mid2(x, t_emb)
            x = self.up1(x, skip3, t_emb)
            x = self.up2(x, skip2, t_emb)
            x = self.up3(x, skip1, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# 6.  EMA  [v0.6: + warmup]
# ──────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            # [v0.6] During warmup, just copy weights — don't average with random init
            for s, m in zip(self.shadow.parameters(), model.parameters()):
                s.data.copy_(m.data)
            return
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ──────────────────────────────────────────────────────────────
# 7.  LR SCHEDULE
# ──────────────────────────────────────────────────────────────

def build_lr_scheduler(optim, warmup_steps: int, total_steps: int, lr_min: float):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, total_steps - warmup_steps), eta_min=lr_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


# ──────────────────────────────────────────────────────────────
# 8.  SAMPLERS  [v0.6: v-prediction aware]
# ──────────────────────────────────────────────────────────────

def _model_to_eps(model, xt, t_batch, sched, prediction_type):
    """[v0.6] Always return ε-prediction regardless of model's native output."""
    pred = model(xt, t_batch)
    if prediction_type == "v":
        return sched.v_to_eps(pred, xt, t_batch)
    return pred  # eps


@torch.no_grad()
def ddpm_sample(model, sched, shape, device, n_steps=None, generator=None, prediction_type="eps"):
    model.eval()
    B = shape[0]
    T = n_steps or sched.T
    x = torch.randn(shape, device=device, generator=generator)

    for t_idx in tqdm(reversed(range(T)), desc="DDPM", total=T, leave=False):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps = _model_to_eps(model, x, t, sched, prediction_type)
        mean, log_var = sched.p_mean_var(eps, x, t)
        if t_idx > 0:
            x = mean + (0.5 * log_var).exp() * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1, 1)


@torch.no_grad()
def ddim_sample(model, sched, shape, device, n_steps=50, eta=0.0, generator=None, prediction_type="eps"):
    model.eval()
    B = shape[0]
    step_ids = torch.linspace(sched.T - 1, 0, n_steps, dtype=torch.long, device=device)
    x = torch.randn(shape, device=device, generator=generator)

    for i, t_idx in enumerate(tqdm(step_ids, desc="DDIM", leave=False)):
        t      = t_idx.expand(B)
        t_prev = step_ids[i + 1] if i + 1 < n_steps else torch.zeros(1, device=device).long()[0]

        eps    = _model_to_eps(model, x, t, sched, prediction_type)
        abar_t = sched.alphas_cumprod[t_idx]
        abar_p = sched.alphas_cumprod[t_prev] if t_prev > 0 else torch.ones(1, device=device)

        x0_pred = ((x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()).clamp(-1, 1)
        sigma   = eta * ((1 - abar_p) / (1 - abar_t) * (1 - abar_t / abar_p)).sqrt()
        noise   = torch.randn_like(x) if eta > 0 else 0.0

        x = abar_p.sqrt() * x0_pred + (1 - abar_p - sigma ** 2).clamp(0).sqrt() * eps + sigma * noise
    return x.clamp(-1, 1)


@torch.no_grad()
def dpm_solver_pp_sample(model, sched, shape, device, n_steps=50, generator=None, prediction_type="eps"):
    """[v0.6] Default n_steps=50 (was 20) — robust for this model size."""
    model.eval()
    B = shape[0]

    ac     = sched.alphas_cumprod.to(device)
    alpha  = ac.sqrt()
    sigma  = (1 - ac).sqrt()
    lam    = torch.log(alpha / sigma)

    seq = torch.linspace(sched.T - 1, 0, n_steps + 1, dtype=torch.long, device=device)

    x = torch.randn(shape, device=device, generator=generator)
    D_prev, h_prev = None, None

    for i in tqdm(range(n_steps), desc="DPM-Solver++", leave=False):
        t_s, t_t = seq[i].item(), seq[i + 1].item()
        t_batch  = torch.full((B,), t_s, device=device, dtype=torch.long)

        eps  = _model_to_eps(model, x, t_batch, sched, prediction_type)
        D_curr = ((x - sigma[t_s] * eps) / alpha[t_s]).clamp(-1, 1)

        h = (lam[t_t] - lam[t_s]).item()

        D_eff = D_curr
        if D_prev is not None and h_prev is not None:
            r    = h_prev / h
            D_eff = (1 + 0.5 / r) * D_curr - (0.5 / r) * D_prev

        coeff_x = alpha[t_t] / alpha[t_s]
        coeff_d = sigma[t_t] * (-torch.expm1(torch.tensor(-h)))
        x = coeff_x * x + coeff_d * D_eff
        D_prev, h_prev = D_curr, h
    return x.clamp(-1, 1)


def sample(model, sched, cfg, device, n=None, generator=None):
    n = n or cfg.n_samples
    shape = (n, 3, cfg.image_size, cfg.image_size)
    pt = cfg.prediction_type
    if cfg.sampler == "ddpm":
        return ddpm_sample(model, sched, shape, device, cfg.sample_steps, generator, pt)
    elif cfg.sampler == "ddim":
        return ddim_sample(model, sched, shape, device, cfg.sample_steps, generator=generator, prediction_type=pt)
    elif cfg.sampler == "dpm_solver":
        return dpm_solver_pp_sample(model, sched, shape, device, cfg.sample_steps, generator, pt)
    else:
        raise ValueError(f"Unknown sampler: {cfg.sampler!r}")


# ──────────────────────────────────────────────────────────────
# 9.  FID  (unchanged)
# ──────────────────────────────────────────────────────────────

def _get_inception(device):
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.aux_logits = False
    m.eval()
    return m.to(device)


@torch.no_grad()
def _extract_features(images, inception, device, batch=64):
    feats = []
    for i in range(0, len(images), batch):
        b = images[i:i + batch].to(device)
        b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
        b = (b + 1) / 2
        feats.append(inception(b).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(real_images, fake_images, device):
    if not SCIPY_AVAILABLE:
        return None
    inception = _get_inception(device)
    f_r = _extract_features(real_images, inception, device)
    f_g = _extract_features(fake_images, inception, device)

    mu_r, sig_r = f_r.mean(0), np.cov(f_r, rowvar=False)
    mu_g, sig_g = f_g.mean(0), np.cov(f_g, rowvar=False)

    diff = mu_r - mu_g
    cov_sqrt, _ = scipy_sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid = float(diff @ diff + np.trace(sig_r + sig_g - 2 * cov_sqrt))
    del inception
    return fid


# ──────────────────────────────────────────────────────────────
# 10. TRAINING LOOP  [v0.6: v-prediction + Min-SNR]
# ──────────────────────────────────────────────────────────────

def train(cfg: Config, device: torch.device) -> dict:
    torch.manual_seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)

    out_dir  = Path(cfg.log_dir) / cfg.phase_name
    ckpt_dir = out_dir / "checkpoints"
    img_dir  = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger(cfg.phase_name)
    log.info(f"Phase: {cfg.phase_name}  [v0.6]")
    log.info(f"Prediction: {cfg.prediction_type}  |  Loss weighting: {cfg.loss_weighting} (γ={cfg.min_snr_gamma})")
    log.info(f"Schedule: {cfg.schedule}" + (f" (shift={cfg.cosine_shift})" if cfg.schedule == "shifted_cosine" else ""))
    log.info(f"Training on REAL face data: {cfg.data_dirs}")
    log.info(json.dumps(asdict(cfg), indent=2))

    train_loader, val_loader, val_set = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = UNet(
        channels=cfg.channels, use_attention=cfg.use_attention,
        attn_resolutions=cfg.attn_resolutions, image_size=cfg.image_size,
        dropout=cfg.dropout, use_checkpoint=cfg.use_checkpoint
    ).to(device)
    ema   = EMA(model, cfg.ema_decay, warmup_steps=cfg.ema_warmup_steps)
    sched = NoiseScheduler(
        cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end, cfg.cosine_shift
    ).to(device)

    n_params = model.param_count
    log.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    log.info(f"Effective batch size: {cfg.batch_size} × {cfg.grad_accum_steps} = {cfg.batch_size * cfg.grad_accum_steps}")
    log.info(f"EMA warmup: {cfg.ema_warmup_steps} steps; then decay={cfg.ema_decay}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optim_steps = (cfg.epochs * len(train_loader)) // cfg.grad_accum_steps
    lr_scheduler = build_lr_scheduler(optim, cfg.warmup_steps, optim_steps, cfg.lr_min)
    log.info(f"LR schedule: warmup {cfg.warmup_steps} steps → cosine over {optim_steps - cfg.warmup_steps} steps")

    scaler = GradScaler("cuda", enabled=cfg.use_fp16)

    history = {
        "phase": cfg.phase_name,
        "prediction_type": cfg.prediction_type,
        "loss_weighting": cfg.loss_weighting,
        "schedule": cfg.schedule,
        "train_loss": [], "val_loss": [],
        "fid": {}, "fid_final": None,
        "sample_time_1000": None, "sample_time_fast": None,
        "n_params": n_params,
    }

    real_fid_imgs: Optional[torch.Tensor] = None

    # Fast intermediate FID uses DPM-Solver 50 steps (robust, ~20× speedup)
    fid_cfg = copy.deepcopy(cfg)
    fid_cfg.sampler = "dpm_solver"
    fid_cfg.sample_steps = 50
    fid_cfg.n_samples = 32

    # ── training ─────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        optim.zero_grad(set_to_none=True)

        for i, x0 in enumerate(tqdm(train_loader,
                                     desc=f"[{cfg.phase_name}] epoch {epoch}/{cfg.epochs}",
                                     leave=False)):
            x0 = x0.to(device)
            B  = x0.size(0)
            t  = torch.randint(0, cfg.T, (B,), device=device, dtype=torch.long)

            with autocast("cuda", enabled=cfg.use_fp16):
                xt, noise = sched.q_sample(x0, t)

                # [v0.6] Build target based on prediction type
                if cfg.prediction_type == "v":
                    target = sched.get_v(x0, noise, t)
                else:  # eps
                    target = noise

                pred = model(xt, t)

                # [v0.6] Per-sample MSE, then Min-SNR-weighted mean
                mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])  # (B,)
                weights = compute_loss_weights(
                    sched.snr, t, cfg.loss_weighting, cfg.min_snr_gamma, cfg.prediction_type
                )
                loss = (mse * weights).mean() / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema.update(model)

            ep_loss  += loss.item() * cfg.grad_accum_steps
            n_batches += 1

        train_loss = ep_loss / n_batches
        history["train_loss"].append(train_loss)

        # ── validation ─────────────────────────────────────
        model.eval()
        v_loss, v_batches = 0.0, 0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                B  = x0.size(0)
                t  = torch.randint(0, cfg.T, (B,), device=device, dtype=torch.long)
                xt, noise = sched.q_sample(x0, t)

                if cfg.prediction_type == "v":
                    target = sched.get_v(x0, noise, t)
                else:
                    target = noise

                with autocast("cuda", enabled=cfg.use_fp16):
                    pred = model(xt, t)
                mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
                weights = compute_loss_weights(
                    sched.snr, t, cfg.loss_weighting, cfg.min_snr_gamma, cfg.prediction_type
                )
                v_loss  += (mse * weights).mean().item()
                v_batches += 1
        val_loss = v_loss / v_batches
        history["val_loss"].append(val_loss)

        current_lr = lr_scheduler.get_last_lr()[0]
        mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                 f"lr={current_lr:.2e}  peak_mem={mem_gb:.1f}GB")

        # ── sample grid ────────────────────────────────────
        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            gen = torch.Generator(device=device).manual_seed(MASTER_SEED)
            imgs = sample(ema.shadow, sched, cfg, device, generator=gen)
            grid = make_grid(imgs * 0.5 + 0.5, nrow=int(cfg.n_samples ** 0.5))
            save_image(grid, img_dir / f"epoch_{epoch:04d}.png")
            log.info(f"Saved sample grid → epoch_{epoch:04d}.png")

        # ── FID ────────────────────────────────────────────
        is_final = (epoch == cfg.epochs)
        do_fid   = (epoch % cfg.fid_every == 0) or is_final
        if do_fid and SCIPY_AVAILABLE:
            if real_fid_imgs is None:
                real_list = []
                for xr in val_loader:
                    real_list.append(xr)
                    if sum(r.size(0) for r in real_list) >= cfg.fid_n_samples:
                        break
                real_fid_imgs = torch.cat(real_list, dim=0)[:cfg.fid_n_samples]
                log.info(f"FID reference: {real_fid_imgs.size(0)} held-out real images")

            sampler_cfg = cfg if is_final else fid_cfg
            tag = "final" if is_final else "fast"
            t_fid_start = time.time()
            fake_list = []
            with torch.no_grad():
                while sum(f.size(0) for f in fake_list) < cfg.fid_n_samples:
                    imgs_fid = sample(ema.shadow, sched, sampler_cfg, device)
                    fake_list.append(imgs_fid.cpu())
            fake_fid_imgs = torch.cat(fake_list, dim=0)[:cfg.fid_n_samples]

            fid = compute_fid(real_fid_imgs, fake_fid_imgs, device)
            history["fid"][epoch] = fid
            if is_final:
                history["fid_final"] = fid
            log.info(f"FID @ epoch {epoch} ({tag}, {sampler_cfg.sampler} "
                     f"{sampler_cfg.sample_steps} steps): {fid:.2f}  "
                     f"[{time.time()-t_fid_start:.0f}s]")

        # ── checkpoint ─────────────────────────────────────
        if epoch % cfg.checkpoint_every == 0 or is_final:
            ckpt = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "ema":       ema.state_dict(),
                "optim":     optim.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "config":    asdict(cfg),
            }
            try:
                torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt")
            except Exception as e:
                log.warning(f"[ckpt] save failed: {e}")

    # ── final gallery ─────────────────────────────────────
    log.info(f"Generating final gallery of {cfg.final_gallery_n} images...")
    gen = torch.Generator(device=device).manual_seed(MASTER_SEED + 1)
    gallery = []
    remaining = cfg.final_gallery_n
    while remaining > 0:
        batch_n = min(cfg.n_samples, remaining)
        tmp_cfg = copy.deepcopy(cfg)
        tmp_cfg.n_samples = batch_n
        imgs = sample(ema.shadow, sched, tmp_cfg, device, generator=gen)
        gallery.append(imgs)
        remaining -= batch_n
    gallery = torch.cat(gallery, dim=0)[:cfg.final_gallery_n]
    gallery_grid = make_grid(gallery * 0.5 + 0.5, nrow=10)
    save_image(gallery_grid, img_dir / "final_gallery.png")
    log.info(f"Saved final gallery → final_gallery.png")

    # ── sampling speed benchmark ───────────────────────────
    shape = (cfg.n_samples, 3, cfg.image_size, cfg.image_size)
    pt = cfg.prediction_type

    t0 = time.time()
    ddpm_sample(ema.shadow, sched, shape, device, n_steps=1000, prediction_type=pt)
    history["sample_time_1000"] = time.time() - t0

    t0 = time.time()
    dpm_solver_pp_sample(ema.shadow, sched, shape, device, n_steps=50, prediction_type=pt)
    history["sample_time_fast"] = time.time() - t0

    log.info(f"DDPM 1000 steps: {history['sample_time_1000']:.1f}s")
    log.info(f"DPM-Solver++ 50 steps: {history['sample_time_fast']:.1f}s")
    log.info(f"Speedup: {history['sample_time_1000']/history['sample_time_fast']:.1f}×")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Training complete.")
    return history


# ──────────────────────────────────────────────────────────────
# 11. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_results(all_histories: List[dict], out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not found — skipping plots.")
        return

    out = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for h in all_histories:
        axes[0].plot(h["train_loss"], label=f"{h['phase']} train")
        axes[0].plot(h["val_loss"],   label=f"{h['phase']} val", linestyle="--")

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Min-SNR-weighted Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend(fontsize=7)

    phase_names, fid_vals = [], []
    for h in all_histories:
        if h["fid_final"] is not None:
            phase_names.append(h["phase"].replace("_", "\n"))
            fid_vals.append(h["fid_final"])

    if fid_vals:
        axes[1].bar(phase_names, fid_vals, color=["steelblue", "seagreen", "darkorange", "crimson"])
        axes[1].set_ylabel("FID ↓"); axes[1].set_title("Final FID per Phase (lower is better)")
        for i, v in enumerate(fid_vals):
            axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out / f"phase_comparison_{ts}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot → {out_path}")


# ──────────────────────────────────────────────────────────────
# 12. ENTRY POINT
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDPM pipeline v0.6 — training dynamics overhaul")
    p.add_argument("--phase",   default="1",
                   help="1 | 2 | 3 | 4 | all | range 2-4 | list 2,3,4")
    p.add_argument("--epochs",  type=int, default=None, help="Override epochs")
    p.add_argument("--batch",   type=int, default=None, help="Override batch size")
    p.add_argument("--accum",   type=int, default=None, help="Override grad accumulation steps")
    p.add_argument("--log_dir", default="./ddpm_runs_v06", help="Output directory")
    p.add_argument("--warmup",  type=int, default=None, help="Override LR warmup steps")
    p.add_argument("--prediction", choices=["eps", "v"], default=None, help="Prediction target override")
    p.add_argument("--weighting", choices=["uniform", "min_snr"], default=None, help="Loss weighting override")
    p.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--no_fp16", action="store_true", help="Disable mixed precision")
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda")  if torch.cuda.is_available()  else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    overrides = {"log_dir": args.log_dir}
    if args.epochs     is not None: overrides["epochs"]            = args.epochs
    if args.batch      is not None: overrides["batch_size"]        = args.batch
    if args.accum      is not None: overrides["grad_accum_steps"]  = args.accum
    if args.warmup     is not None: overrides["warmup_steps"]      = args.warmup
    if args.prediction is not None: overrides["prediction_type"]   = args.prediction
    if args.weighting  is not None: overrides["loss_weighting"]    = args.weighting
    if args.checkpoint:              overrides["use_checkpoint"]   = True
    if args.no_fp16:                 overrides["use_fp16"]         = False

    if args.phase == "all":
        phases = [1, 2, 3, 4]
    elif "-" in args.phase:
        start, end = args.phase.split("-")
        phases = list(range(int(start), int(end) + 1))
    elif "," in args.phase:
        phases = [int(p) for p in args.phase.split(",")]
    else:
        phases = [int(args.phase)]

    all_histories = []
    for ph in phases:
        cfg = get_phase_config(ph, **overrides)
        print(f"\n{'='*70}")
        print(f"  Phase {ph}: {cfg.phase_name}  [v0.6 — training dynamics]")
        print(f"  prediction={cfg.prediction_type}  loss_weighting={cfg.loss_weighting}(γ={cfg.min_snr_gamma})")
        print(f"  schedule={cfg.schedule}" + (f" (shift={cfg.cosine_shift})" if cfg.schedule == "shifted_cosine" else ""))
        print(f"  sampler={cfg.sampler}({cfg.sample_steps})  attention={cfg.use_attention}")
        print(f"  batch={cfg.batch_size}×{cfg.grad_accum_steps}={cfg.batch_size*cfg.grad_accum_steps}")
        print(f"  ema_warmup={cfg.ema_warmup_steps}  seed={MASTER_SEED}")
        print(f"{'='*70}\n")
        h = train(cfg, device)
        all_histories.append(h)

    if len(all_histories) > 1:
        plot_results(all_histories, args.log_dir)

    print("\n" + "=" * 60)
    print(f"{'Phase':<30}  {'Final FID':>10}  {'1000-step(s)':>12}  {'50-step(s)':>10}")
    print("-" * 60)
    for h in all_histories:
        fid_str = f"{h['fid_final']:.2f}" if h["fid_final"] is not None else "n/a"
        t1000   = f"{h['sample_time_1000']:.1f}" if h["sample_time_1000"] else "n/a"
        t50     = f"{h['sample_time_fast']:.1f}"  if h["sample_time_fast"]  else "n/a"
        print(f"{h['phase']:<30}  {fid_str:>10}  {t1000:>12}  {t50:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()