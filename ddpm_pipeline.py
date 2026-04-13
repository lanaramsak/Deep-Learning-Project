"""
DDPM Face Generation — DeepFakeFace (DFF) Dataset
==================================================
Four documented improvement phases:
  Phase 1 – Baseline U-Net, no attention, linear schedule, DDPM 1000 steps
  Phase 2 – + Self-attention at 16×16 bottleneck
  Phase 3 – + Cosine noise schedule (replace linear)
  Phase 4 – + DPM-Solver++(2M) sampler (20 steps vs 1000)

Usage:
  python ddpm_pipeline.py --phase 1          # run single phase
  python ddpm_pipeline.py --phase all        # run all four phases sequentially
  python ddpm_pipeline.py --phase 4 --epochs 5  # quick test
"""

import os, sys, math, copy, json, time, argparse, logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
    print("[warn] scipy not found — FID will be skipped. Install with: pip install scipy")

# ──────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────

BASE_DIR = Path(os.environ.get("DDPM_BASE_DIR", str(Path(__file__).parent.parent)))

DATA_DIRS = [
    str(BASE_DIR / "inpainting"),
    str(BASE_DIR / "insight"),
    str(BASE_DIR / "text2img"),
]


@dataclass
class Config:
    # ── data ──────────────────────────────────────────────────
    data_dirs:    List[str] = field(default_factory=lambda: DATA_DIRS)
    image_size:   int       = 64
    val_split:    float     = 0.10
    num_workers:  int       = 4

    # ── model ─────────────────────────────────────────────────
    channels:     List[int] = field(default_factory=lambda: [64, 128, 256])
    use_attention: bool     = False   # Phase 1: off; Phase 2+: on

    # ── noise schedule ────────────────────────────────────────
    T:            int       = 1000
    schedule:     str       = "linear"   # "linear" | "cosine"
    beta_start:   float     = 1e-4
    beta_end:     float     = 2e-2

    # ── training ──────────────────────────────────────────────
    epochs:       int       = 100
    batch_size:   int       = 64
    lr:           float     = 2e-4
    ema_decay:    float     = 0.9999
    use_fp16:     bool      = True
    grad_clip:    float     = 1.0

    # ── sampling ──────────────────────────────────────────────
    sampler:      str       = "ddpm"    # "ddpm" | "ddim" | "dpm_solver"
    sample_steps: int       = 1000     # inference steps
    n_samples:    int       = 16       # for visualisation grid

    # ── logging ───────────────────────────────────────────────
    log_dir:         str  = "./ddpm_runs"
    sample_every:    int  = 10
    checkpoint_every: int = 10
    phase_name:      str  = "phase1_baseline"
    fid_n_samples:   int  = 2048   # generated samples for FID


# Four phase configs — each builds on the previous
def get_phase_config(phase: int, **overrides) -> Config:
    cfg = Config()
    if phase >= 2:
        cfg.use_attention = True
        cfg.phase_name = "phase2_attention"
    if phase >= 3:
        cfg.schedule = "cosine"
        cfg.phase_name = "phase3_cosine"
    if phase >= 4:
        cfg.sampler = "dpm_solver"
        cfg.sample_steps = 20
        cfg.phase_name = "phase4_dpm_solver"
    if phase == 1:
        cfg.phase_name = "phase1_baseline"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ──────────────────────────────────────────────────────────────
# 2.  DATASET
# ──────────────────────────────────────────────────────────────

class FakeFaceDataset(Dataset):
    """Loads fake face images from the three DFF generators."""

    def __init__(self, data_dirs: List[str], image_size: int = 64, augment: bool = True):
        self.paths: List[Path] = []
        for d in data_dirs:
            p = Path(d)
            if not p.exists():
                print(f"[warn] directory not found: {p}")
                continue
            self.paths.extend(p.rglob("*.jpg"))
            self.paths.extend(p.rglob("*.png"))
        print(f"Dataset: {len(self.paths):,} images from {len(data_dirs)} dirs")

        tf_list: list = []
        if augment:
            tf_list.append(transforms.RandomHorizontalFlip())
        tf_list += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ]
        self.transform = transforms.Compose(tf_list)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_dataloaders(cfg: Config):
    full = FakeFaceDataset(cfg.data_dirs, cfg.image_size, augment=True)
    n_val  = int(len(full) * cfg.val_split)
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Val set: no augmentation
    val_set.dataset = FakeFaceDataset(cfg.data_dirs, cfg.image_size, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────
# 3.  NOISE SCHEDULER
# ──────────────────────────────────────────────────────────────

class NoiseScheduler:
    """
    Implements DDPM noise schedule and closed-form forward process
    q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)·x_0, (1−ᾱ_t)·I).
    """

    def __init__(self, T: int = 1000, schedule: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.T = T

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        elif schedule == "cosine":
            # Nichol & Dhariwal 2021 — improved DDPM
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = (1.0 - alphas_bar[1:] / alphas_bar[:-1]).float()
            betas = betas.clamp(1e-4, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas              = betas
        self.alphas             = alphas
        self.alphas_cumprod     = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Forward process helpers
        self.sqrt_alphas_cumprod      = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_log_var  = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_c1  = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        self.posterior_mean_c2  = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)

    def to(self, device: torch.device) -> "NoiseScheduler":
        for attr in vars(self):
            v = getattr(self, attr)
            if isinstance(v, torch.Tensor):
                setattr(self, attr, v.to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form forward process — add noise at any timestep in one shot."""
        if noise is None:
            noise = torch.randn_like(x0)
        s  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s * x0 + s1 * noise, noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor,
                   eps: torch.Tensor) -> torch.Tensor:
        """Recover x0 estimate from noise prediction."""
        s  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (xt - s1 * eps) / s

    def p_mean_var(self, eps: torch.Tensor, xt: torch.Tensor,
                   t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDPM reverse-process posterior mean and log-variance."""
        x0_pred = self.predict_x0(xt, t, eps).clamp(-1, 1)
        mean = (
            self.posterior_mean_c1[t][:, None, None, None] * x0_pred
            + self.posterior_mean_c2[t][:, None, None, None] * xt
        )
        log_var = self.posterior_log_var[t][:, None, None, None]
        return mean, log_var


# ──────────────────────────────────────────────────────────────
# 4.  U-NET
# ──────────────────────────────────────────────────────────────

def sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings (Vaswani et al.)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]      # (B, half)
    return torch.cat([args.cos(), args.sin()], dim=-1)   # (B, dim)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1  = nn.GroupNorm(32, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(32, out_ch)
        self.drop   = nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions."""

    def __init__(self, ch: int, n_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, t_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res2(self.res1(x, t), t)
        return self.down(x), x   # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, t_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.up(x), skip], dim=1)
        return self.res2(self.res1(x, t), t)


class UNet(nn.Module):
    """
    U-Net denoiser for 64×64 images.

    Encoder:   64×64 → 32×32 → 16×16
    Bottleneck: 16×16  (optional self-attention here)
    Decoder:   16×16 → 32×32 → 64×64

    channels = [C0=64, C1=128, C2=256]
    """

    def __init__(self, in_ch: int = 3, channels: List[int] = None,
                 use_attention: bool = True):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        C0, C1, C2 = channels
        t_dim = C0 * 4   # timestep embedding dimension (256)

        self.time_mlp = nn.Sequential(
            nn.Linear(C0, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, C0, 3, padding=1)

        # Encoder
        self.down1 = DownBlock(C0, C1, t_dim)   # 64→32, skip (B,C1,64,64)
        self.down2 = DownBlock(C1, C2, t_dim)   # 32→16, skip (B,C2,32,32)

        # Bottleneck at 16×16
        self.mid1 = ResBlock(C2, C2, t_dim)
        self.mid_attn = SelfAttention(C2) if use_attention else nn.Identity()
        self.mid2 = ResBlock(C2, C2, t_dim)

        # Decoder
        self.up1 = UpBlock(C2, C2, C1, t_dim)   # 16→32
        self.up2 = UpBlock(C1, C1, C0, t_dim)   # 32→64

        self.out_norm = nn.GroupNorm(32, C0)
        self.out_conv = nn.Conv2d(C0, in_ch, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_emb(t, self.time_mlp[0].in_features))

        x = self.in_conv(x)                       # (B,C0,64,64)
        x, skip1 = self.down1(x, t_emb)           # (B,C1,32,32)
        x, skip2 = self.down2(x, t_emb)           # (B,C2,16,16)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        x = self.up1(x, skip2, t_emb)             # (B,C1,32,32)
        x = self.up2(x, skip1, t_emb)             # (B,C0,64,64)

        return self.out_conv(F.silu(self.out_norm(x)))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# 5.  EMA
# ──────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights for stable sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ──────────────────────────────────────────────────────────────
# 6.  SAMPLERS
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model: nn.Module, sched: NoiseScheduler,
                shape: Tuple, device: torch.device,
                n_steps: Optional[int] = None) -> torch.Tensor:
    """
    DDPM ancestral sampling (Algorithm 2, Ho et al. 2020).
    Default: n_steps = T = 1000.
    """
    model.eval()
    B = shape[0]
    T = n_steps or sched.T
    x = torch.randn(shape, device=device)

    for t_idx in tqdm(reversed(range(T)), desc="DDPM", total=T, leave=False):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps = model(x, t)
        mean, log_var = sched.p_mean_var(eps, x, t)
        if t_idx > 0:
            x = mean + (0.5 * log_var).exp() * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1, 1)


@torch.no_grad()
def ddim_sample(model: nn.Module, sched: NoiseScheduler,
                shape: Tuple, device: torch.device,
                n_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
    """
    DDIM deterministic sampler (Song et al. 2020).
    eta=0 → fully deterministic; eta=1 → DDPM-equivalent stochasticity.
    """
    model.eval()
    B = shape[0]
    step_ids = torch.linspace(sched.T - 1, 0, n_steps, dtype=torch.long, device=device)
    x = torch.randn(shape, device=device)

    for i, t_idx in enumerate(tqdm(step_ids, desc="DDIM", leave=False)):
        t      = t_idx.expand(B)
        t_prev = step_ids[i + 1] if i + 1 < n_steps else torch.zeros(1, device=device).long()[0]

        eps    = model(x, t)
        abar_t = sched.alphas_cumprod[t_idx]
        abar_p = sched.alphas_cumprod[t_prev] if t_prev > 0 else torch.ones(1, device=device)

        x0_pred = ((x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()).clamp(-1, 1)
        sigma   = eta * ((1 - abar_p) / (1 - abar_t) * (1 - abar_t / abar_p)).sqrt()
        noise   = torch.randn_like(x) if eta > 0 else 0.0

        x = abar_p.sqrt() * x0_pred + (1 - abar_p - sigma ** 2).clamp(0).sqrt() * eps + sigma * noise

    return x.clamp(-1, 1)


@torch.no_grad()
def dpm_solver_pp_sample(model: nn.Module, sched: NoiseScheduler,
                          shape: Tuple, device: torch.device,
                          n_steps: int = 20) -> torch.Tensor:
    """
    DPM-Solver++(2M) — Algorithm 2 (Lu et al. 2022).

    Uses second-order multistep updates in λ-space:
      λ_t = log(α_t / σ_t)   where α_t = sqrt(ᾱ_t), σ_t = sqrt(1−ᾱ_t)

    Update rule:
      1st step  : x_{t+1} = (α_next/α_curr)·x_t + σ_next·(1−e^{−h})·D_curr
      2nd+ steps: D_eff = (1+0.5/r)·D_curr − (0.5/r)·D_prev  (r = h_prev/h)
                  x_{t+1} = (α_next/α_curr)·x_t + σ_next·(1−e^{−h})·D_eff
    """
    model.eval()
    B = shape[0]

    # Precompute λ schedule
    ac     = sched.alphas_cumprod.to(device)
    alpha  = ac.sqrt()                        # sqrt(ᾱ_t)
    sigma  = (1 - ac).sqrt()                  # sqrt(1−ᾱ_t)
    lam    = torch.log(alpha / sigma)         # λ_t

    # Timestep sequence: uniform in index space (T-1 → 0)
    seq = torch.linspace(sched.T - 1, 0, n_steps + 1, dtype=torch.long, device=device)

    x = torch.randn(shape, device=device)
    D_prev, h_prev = None, None

    for i in tqdm(range(n_steps), desc="DPM-Solver++", leave=False):
        t_s, t_t = seq[i].item(), seq[i + 1].item()
        t_batch  = torch.full((B,), t_s, device=device, dtype=torch.long)

        # ε-prediction → x̂0
        eps  = model(x, t_batch)
        D_curr = ((x - sigma[t_s] * eps) / alpha[t_s]).clamp(-1, 1)

        h = (lam[t_t] - lam[t_s]).item()        # > 0 (moving toward clean)

        D_eff = D_curr
        if D_prev is not None and h_prev is not None:
            r    = h_prev / h
            D_eff = (1 + 0.5 / r) * D_curr - (0.5 / r) * D_prev

        # x0-prediction form update
        coeff_x = alpha[t_t] / alpha[t_s]
        coeff_d = sigma[t_t] * (-torch.expm1(torch.tensor(-h)))  # σ_{t+1}·(1−e^{−h})
        x = coeff_x * x + coeff_d * D_eff

        D_prev, h_prev = D_curr, h

    return x.clamp(-1, 1)


def sample(model: nn.Module, sched: NoiseScheduler, cfg: Config,
           device: torch.device) -> torch.Tensor:
    """Dispatch to the configured sampler."""
    shape = (cfg.n_samples, 3, cfg.image_size, cfg.image_size)
    if cfg.sampler == "ddpm":
        return ddpm_sample(model, sched, shape, device, cfg.sample_steps)
    elif cfg.sampler == "ddim":
        return ddim_sample(model, sched, shape, device, cfg.sample_steps)
    elif cfg.sampler == "dpm_solver":
        return dpm_solver_pp_sample(model, sched, shape, device, cfg.sample_steps)
    else:
        raise ValueError(f"Unknown sampler: {cfg.sampler!r}")


# ──────────────────────────────────────────────────────────────
# 7.  FID
# ──────────────────────────────────────────────────────────────

def _get_inception(device: torch.device) -> nn.Module:
    """Load InceptionV3 in feature-extraction mode."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    m.fc = nn.Identity()   # return 2048-dim pool features
    m.aux_logits = False
    m.eval()
    return m.to(device)


@torch.no_grad()
def _extract_features(images: torch.Tensor, inception: nn.Module,
                       device: torch.device, batch: int = 64) -> np.ndarray:
    feats = []
    for i in range(0, len(images), batch):
        b = images[i:i + batch].to(device)
        b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
        b = (b + 1) / 2   # [-1,1] → [0,1]
        feats.append(inception(b).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor,
                device: torch.device) -> Optional[float]:
    """
    Compute FID between two sets of images in [-1,1] range.
    Returns None if scipy is not installed.
    """
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
# 8.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────

def train(cfg: Config, device: torch.device) -> dict:
    """
    Full training run for one phase.
    Returns a dict of logged metrics (loss curve, FID scores).
    """
    # ── setup ───────────────────────────────────────────────
    out_dir = Path(cfg.log_dir) / cfg.phase_name
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
    log.info(f"Phase: {cfg.phase_name}")
    log.info(json.dumps(asdict(cfg), indent=2))

    # ── data ────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── model ───────────────────────────────────────────────
    model = UNet(channels=cfg.channels, use_attention=cfg.use_attention).to(device)
    ema   = EMA(model, cfg.ema_decay)
    sched = NoiseScheduler(cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end).to(device)

    log.info(f"Model parameters: {model.param_count:,}")
    log.info(f"Schedule: {cfg.schedule} | Attention: {cfg.use_attention}")

    # ── optimiser & scaler ──────────────────────────────────
    optim  = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.use_fp16)

    # ── metrics storage ─────────────────────────────────────
    history = {
        "phase": cfg.phase_name,
        "train_loss": [], "val_loss": [],
        "fid": {}, "sample_time_1000": None, "sample_time_fast": None
    }

    # Collect val images once for FID reference
    real_fid_imgs: Optional[torch.Tensor] = None

    # ── training ─────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0

        for x0 in tqdm(train_loader, desc=f"[{cfg.phase_name}] epoch {epoch}/{cfg.epochs}",
                        leave=False):
            x0 = x0.to(device)
            B  = x0.size(0)
            t  = torch.randint(0, cfg.T, (B,), device=device, dtype=torch.long)

            with autocast(enabled=cfg.use_fp16):
                xt, noise = sched.q_sample(x0, t)
                pred_noise = model(xt, t)
                loss = F.mse_loss(pred_noise, noise)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            ema.update(model)

            ep_loss  += loss.item()
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
                with autocast(enabled=cfg.use_fp16):
                    pred = model(xt, t)
                v_loss  += F.mse_loss(pred, noise).item()
                v_batches += 1
        val_loss = v_loss / v_batches
        history["val_loss"].append(val_loss)

        log.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ── sample grid ────────────────────────────────────
        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            imgs = sample(ema.shadow, sched, cfg, device)
            grid = make_grid(imgs * 0.5 + 0.5, nrow=int(cfg.n_samples ** 0.5))
            save_image(grid, img_dir / f"epoch_{epoch:04d}.png")
            log.info(f"Saved sample grid → {img_dir / f'epoch_{epoch:04d}.png'}")

            # FID at final epoch and every sample_every epochs
            if SCIPY_AVAILABLE:
                if real_fid_imgs is None:
                    # Collect cfg.fid_n_samples real images once
                    real_list = []
                    for xr in val_loader:
                        real_list.append(xr)
                        if sum(r.size(0) for r in real_list) >= cfg.fid_n_samples:
                            break
                    real_fid_imgs = torch.cat(real_list, dim=0)[:cfg.fid_n_samples]

                # Generate cfg.fid_n_samples fake images
                fake_list = []
                with torch.no_grad():
                    while sum(f.size(0) for f in fake_list) < cfg.fid_n_samples:
                        imgs_fid = sample(ema.shadow, sched, cfg, device)
                        fake_list.append(imgs_fid.cpu())
                fake_fid_imgs = torch.cat(fake_list, dim=0)[:cfg.fid_n_samples]

                fid = compute_fid(real_fid_imgs, fake_fid_imgs, device)
                history["fid"][epoch] = fid
                log.info(f"FID @ epoch {epoch}: {fid:.2f}")

        # ── checkpoint ─────────────────────────────────────
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            ckpt = {
                "epoch":      epoch,
                "model":      model.state_dict(),
                "ema":        ema.state_dict(),
                "optim":      optim.state_dict(),
                "config":     asdict(cfg),
            }
            torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt")

    # ── measure sampling speed ─────────────────────────────
    shape = (cfg.n_samples, 3, cfg.image_size, cfg.image_size)

    t0 = time.time()
    ddpm_sample(ema.shadow, sched, shape, device, n_steps=1000)
    history["sample_time_1000"] = time.time() - t0

    t0 = time.time()
    dpm_solver_pp_sample(ema.shadow, sched, shape, device, n_steps=20)
    history["sample_time_fast"] = time.time() - t0

    log.info(f"DDPM 1000 steps: {history['sample_time_1000']:.1f}s")
    log.info(f"DPM-Solver++ 20 steps: {history['sample_time_fast']:.1f}s")
    log.info(f"Speedup: {history['sample_time_1000']/history['sample_time_fast']:.1f}×")

    # ── save history ───────────────────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Training complete.")
    return history


# ──────────────────────────────────────────────────────────────
# 9.  PLOTTING (loss curves + FID comparison)
# ──────────────────────────────────────────────────────────────

def plot_results(all_histories: List[dict], out_dir: str = "./ddpm_runs"):
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

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend(fontsize=7)

    # FID bar chart (final epoch)
    phase_names, fid_vals = [], []
    for h in all_histories:
        if h["fid"]:
            last_epoch = max(h["fid"].keys(), key=int)
            phase_names.append(h["phase"].replace("_", "\n"))
            fid_vals.append(h["fid"][last_epoch])

    if fid_vals:
        axes[1].bar(phase_names, fid_vals, color=["steelblue", "seagreen", "darkorange", "crimson"])
        axes[1].set_ylabel("FID ↓"); axes[1].set_title("FID per Phase (lower is better)")
        for i, v in enumerate(fid_vals):
            axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out / "phase_comparison.png", dpi=150)
    plt.close()
    print(f"Saved comparison plot → {out / 'phase_comparison.png'}")


# ──────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDPM pipeline — DFF fake faces")
    p.add_argument("--phase",   default="1",
                   help="1 | 2 | 3 | 4 | all  (default: 1)")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override number of epochs (e.g. 5 for quick test)")
    p.add_argument("--batch",   type=int, default=None, help="Override batch size")
    p.add_argument("--log_dir", default="./ddpm_runs", help="Output directory")
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
    if args.epochs  is not None: overrides["epochs"]     = args.epochs
    if args.batch   is not None: overrides["batch_size"] = args.batch
    if args.no_fp16:              overrides["use_fp16"]  = False

    phases = [1, 2, 3, 4] if args.phase == "all" else [int(args.phase)]
    all_histories = []

    for ph in phases:
        cfg = get_phase_config(ph, **overrides)
        print(f"\n{'='*60}")
        print(f"  Phase {ph}: {cfg.phase_name}")
        print(f"  attention={cfg.use_attention}  schedule={cfg.schedule}  sampler={cfg.sampler}({cfg.sample_steps})")
        print(f"{'='*60}\n")
        h = train(cfg, device)
        all_histories.append(h)

    if len(all_histories) > 1:
        plot_results(all_histories, args.log_dir)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Phase':<30}  {'Final FID':>10}  {'1000-step(s)':>12}  {'20-step(s)':>10}")
    print("-" * 60)
    for h in all_histories:
        fid_str = f"{list(h['fid'].values())[-1]:.2f}" if h["fid"] else "n/a"
        t1000   = f"{h['sample_time_1000']:.1f}" if h["sample_time_1000"] else "n/a"
        t20     = f"{h['sample_time_fast']:.1f}"  if h["sample_time_fast"]  else "n/a"
        print(f"{h['phase']:<30}  {fid_str:>10}  {t1000:>12}  {t20:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
