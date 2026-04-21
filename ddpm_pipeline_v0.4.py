"""
DDPM Face Generation — Real Face Training (v0.4)
=================================================
Goal: Train a DDPM on REAL face images (wiki dataset) to learn the true
      face distribution, then generate new synthetic ("deepfake") faces.

Four documented improvement phases (each builds on the previous):
  Phase 1 – Baseline U-Net, no attention, linear schedule, DDPM 1000 steps
  Phase 2 – + Self-attention at 16×16 bottleneck
  Phase 3 – + Cosine noise schedule (replace linear)
  Phase 4 – + DPM-Solver++(2M) sampler (20 steps vs 1000)

Changes vs v0.3:
  [FIX] LR warmup (500 steps linear ramp) + CosineAnnealing → prevents
        attention collapse observed in Phase 2 of v0.3 (green-only samples)
  [FIX] FID reference uses fixed seed — all phases compare against IDENTICAL
        real images, making FID scores directly comparable between phases
  [FIX] Intermediate FIDs use DPM-Solver++ 20 steps (fast) — saves ~1h per
        phase. Final FID still uses configured sampler for fair comparison
  [FIX] FID only at epochs 25/50/75/100 instead of every 10 epochs
  [FIX] Final galleries: 100 generated samples saved at end of each phase
  [FIX] torch.amp instead of deprecated torch.cuda.amp
  [FIX] Master seed for model init + sampling (reproducibility)

Kept from v0.3:
  - Trains on REAL wiki images
  - Attention only at 16×16 bottleneck (OOM-safe)
  - AdamW + val_set bug fix

Usage on HPC:
  sbatch -p normal --qos gpu_batch --gres=gpu:1 --time=16:00:00 \
    --job-name "v04_p2" \
    --wrap "python3 ~/ddpm_pipeline_v0.4.py --phase 2 --epochs 100 --log_dir /data/01/up202512956/ddpm_runs_v04"
"""

import os, sys, math, copy, json, time, argparse, logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# [v0.4] New non-deprecated API
from torch.amp import GradScaler, autocast
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

BASE_DIR = Path(os.environ.get("DDPM_BASE_DIR", "/data/01/up202402612/data"))
REAL_DIR = BASE_DIR / "wiki"

# [v0.4] Global master seed — controls model init, sampling noise, and
# the FID reference selection across ALL phases for direct comparability
MASTER_SEED = 42


@dataclass
class Config:
    # ── data ──────────────────────────────────────────────────
    data_dirs:    List[str] = field(default_factory=lambda: [str(REAL_DIR)])
    image_size:   int       = 64
    val_split:    float     = 0.10
    num_workers:  int       = 4

    # ── model ─────────────────────────────────────────────────
    channels:     List[int] = field(default_factory=lambda: [64, 128, 256])
    use_attention: bool     = False

    # ── noise schedule ────────────────────────────────────────
    T:            int       = 1000
    schedule:     str       = "linear"
    beta_start:   float     = 1e-4
    beta_end:     float     = 2e-2

    # ── training ──────────────────────────────────────────────
    epochs:         int   = 100
    batch_size:     int   = 64
    lr:             float = 2e-4
    lr_min:         float = 1e-6
    warmup_steps:   int   = 500       # [v0.4] LR warmup — fixes attention collapse
    weight_decay:   float = 1e-4
    ema_decay:      float = 0.9999
    use_fp16:       bool  = True
    grad_clip:      float = 1.0

    # ── sampling ──────────────────────────────────────────────
    sampler:      str  = "ddpm"
    sample_steps: int  = 1000
    n_samples:    int  = 16

    # ── logging ───────────────────────────────────────────────
    log_dir:          str  = "./ddpm_runs_v04"
    sample_every:     int  = 10        # grid saves every 10 epochs
    fid_every:        int  = 25        # [v0.4] FID less frequently (was 10)
    checkpoint_every: int  = 25        # [v0.4] save checkpoints less often
    phase_name:       str  = "phase1_baseline"
    fid_n_samples:    int  = 2048

    # [v0.4] Final gallery: many samples saved at end of training for reporting
    final_gallery_n:  int  = 100


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

class FaceDataset(Dataset):
    def __init__(self, data_dirs: List[str], image_size: int = 64, augment: bool = True):
        self.image_size = image_size
        self.paths: List[Path] = []
        for d in data_dirs:
            p = Path(d)
            if not p.exists():
                print(f"[warn] directory not found: {p}")
                continue
            self.paths.extend(p.rglob("*.jpg"))
            self.paths.extend(p.rglob("*.png"))
        # [v0.4] Sort for deterministic ordering across runs
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
    # [v0.4] Split uses MASTER_SEED — identical train/val split across phases
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
# 3.  NOISE SCHEDULER  (unchanged from v0.3)
# ──────────────────────────────────────────────────────────────

class NoiseScheduler:
    def __init__(self, T=1000, schedule="linear", beta_start=1e-4, beta_end=2e-2):
        self.T = T

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        elif schedule == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = (1.0 - alphas_bar[1:] / alphas_bar[:-1]).float().clamp(1e-4, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas              = betas
        self.alphas             = alphas
        self.alphas_cumprod     = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod      = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_log_var  = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_c1  = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        self.posterior_mean_c2  = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)

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
# 4.  U-NET  (unchanged from v0.3)
# ──────────────────────────────────────────────────────────────

def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1  = nn.GroupNorm(32, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(32, out_ch)
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
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, t_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res2(self.res1(x, t), t)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, t_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_dim)

    def forward(self, x, skip, t):
        x = torch.cat([self.up(x), skip], dim=1)
        return self.res2(self.res1(x, t), t)


class UNet(nn.Module):
    def __init__(self, in_ch=3, channels=None, use_attention=True):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        C0, C1, C2 = channels
        t_dim = C0 * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(C0, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.in_conv = nn.Conv2d(in_ch, C0, 3, padding=1)

        self.down1 = DownBlock(C0, C1, t_dim)
        self.down2 = DownBlock(C1, C2, t_dim)

        self.mid1 = ResBlock(C2, C2, t_dim)
        self.mid_attn = SelfAttention(C2, n_heads=8) if use_attention else nn.Identity()
        self.mid2 = ResBlock(C2, C2, t_dim)

        self.up1 = UpBlock(C2, C2, C1, t_dim)
        self.up2 = UpBlock(C1, C1, C0, t_dim)

        self.out_norm = nn.GroupNorm(32, C0)
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
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# 5.  EMA
# ──────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ──────────────────────────────────────────────────────────────
# 6.  LR SCHEDULE  [v0.4 NEW]
# ──────────────────────────────────────────────────────────────

def build_lr_scheduler(optim, warmup_steps: int, total_steps: int, lr_min: float):
    """
    Linear warmup for `warmup_steps` followed by cosine annealing to `lr_min`.

    This is critical for DDPM training with attention — without warmup, the
    attention layers can collapse to uninformative patterns in the first few
    updates (observed in v0.3 as all-green samples).
    """
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
# 7.  SAMPLERS  (unchanged from v0.3)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model, sched, shape, device, n_steps=None, generator=None):
    model.eval()
    B = shape[0]
    T = n_steps or sched.T
    x = torch.randn(shape, device=device, generator=generator)

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
def ddim_sample(model, sched, shape, device, n_steps=50, eta=0.0, generator=None):
    model.eval()
    B = shape[0]
    step_ids = torch.linspace(sched.T - 1, 0, n_steps, dtype=torch.long, device=device)
    x = torch.randn(shape, device=device, generator=generator)

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
def dpm_solver_pp_sample(model, sched, shape, device, n_steps=20, generator=None):
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

        eps  = model(x, t_batch)
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
    """Dispatch to the configured sampler. [v0.4] supports n override + generator."""
    n = n or cfg.n_samples
    shape = (n, 3, cfg.image_size, cfg.image_size)
    if cfg.sampler == "ddpm":
        return ddpm_sample(model, sched, shape, device, cfg.sample_steps, generator)
    elif cfg.sampler == "ddim":
        return ddim_sample(model, sched, shape, device, cfg.sample_steps, generator=generator)
    elif cfg.sampler == "dpm_solver":
        return dpm_solver_pp_sample(model, sched, shape, device, cfg.sample_steps, generator)
    else:
        raise ValueError(f"Unknown sampler: {cfg.sampler!r}")


# ──────────────────────────────────────────────────────────────
# 8.  FID
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
# 9.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────

def train(cfg: Config, device: torch.device) -> dict:
    # [v0.4] Set seeds deterministically at start of each phase
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
    log.info(f"Phase: {cfg.phase_name}  [v0.4]")
    log.info(f"Training on REAL face data: {cfg.data_dirs}")
    log.info(json.dumps(asdict(cfg), indent=2))

    # ── data ────────────────────────────────────────────────
    train_loader, val_loader, val_set = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── model ───────────────────────────────────────────────
    model = UNet(channels=cfg.channels, use_attention=cfg.use_attention).to(device)
    ema   = EMA(model, cfg.ema_decay)
    sched = NoiseScheduler(cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end).to(device)

    log.info(f"Model parameters: {model.param_count:,}")
    log.info(f"Schedule: {cfg.schedule} | Attention: {cfg.use_attention}")

    # ── optimiser, scheduler & scaler ──────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # [v0.4] LR = warmup + cosine (per-step, not per-epoch)
    total_steps  = cfg.epochs * len(train_loader)
    lr_scheduler = build_lr_scheduler(optim, cfg.warmup_steps, total_steps, cfg.lr_min)
    log.info(f"LR schedule: warmup {cfg.warmup_steps} steps → cosine over {total_steps - cfg.warmup_steps} steps")

    scaler = GradScaler("cuda", enabled=cfg.use_fp16)

    history = {
        "phase": cfg.phase_name,
        "train_loss": [], "val_loss": [],
        "fid": {}, "fid_final": None,
        "sample_time_1000": None, "sample_time_fast": None
    }

    # ── FID reference (real images) — SAME for every phase via MASTER_SEED
    real_fid_imgs: Optional[torch.Tensor] = None

    # [v0.4] Fast sampler for intermediate FID (DPM-Solver++ 20 steps)
    # Saves ~10-15 min per FID check vs DDPM 1000 steps
    fid_cfg = Config()
    fid_cfg.sampler = "dpm_solver"
    fid_cfg.sample_steps = 20
    fid_cfg.n_samples = 64  # larger batch for faster FID accumulation
    fid_cfg.image_size = cfg.image_size

    # ── training ─────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0

        for x0 in tqdm(train_loader, desc=f"[{cfg.phase_name}] epoch {epoch}/{cfg.epochs}",
                        leave=False):
            x0 = x0.to(device)
            B  = x0.size(0)
            t  = torch.randint(0, cfg.T, (B,), device=device, dtype=torch.long)

            with autocast("cuda", enabled=cfg.use_fp16):
                xt, noise = sched.q_sample(x0, t)
                pred_noise = model(xt, t)
                loss = F.mse_loss(pred_noise, noise)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            # [v0.4] Step LR per iteration (warmup needs this)
            lr_scheduler.step()
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
                with autocast("cuda", enabled=cfg.use_fp16):
                    pred = model(xt, t)
                v_loss  += F.mse_loss(pred, noise).item()
                v_batches += 1
        val_loss = v_loss / v_batches
        history["val_loss"].append(val_loss)

        current_lr = lr_scheduler.get_last_lr()[0]
        log.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.2e}")

        # ── sample grid (every sample_every epochs) ────────
        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            # Fixed seed for grid — samples are comparable epoch-to-epoch
            gen = torch.Generator(device=device).manual_seed(MASTER_SEED)
            imgs = sample(ema.shadow, sched, cfg, device, generator=gen)
            grid = make_grid(imgs * 0.5 + 0.5, nrow=int(cfg.n_samples ** 0.5))
            save_image(grid, img_dir / f"epoch_{epoch:04d}.png")
            log.info(f"Saved sample grid → epoch_{epoch:04d}.png")

        # ── FID (only every fid_every epochs) ──────────────
        is_final = (epoch == cfg.epochs)
        do_fid   = (epoch % cfg.fid_every == 0) or is_final
        if do_fid and SCIPY_AVAILABLE:
            # [v0.4] Build FID reference ONCE, deterministically — identical
            # across all phases because the val split uses MASTER_SEED and
            # val_loader here iterates in order (shuffle=False)
            if real_fid_imgs is None:
                real_list = []
                for xr in val_loader:
                    real_list.append(xr)
                    if sum(r.size(0) for r in real_list) >= cfg.fid_n_samples:
                        break
                real_fid_imgs = torch.cat(real_list, dim=0)[:cfg.fid_n_samples]
                log.info(f"FID reference: {real_fid_imgs.size(0)} held-out real images (deterministic)")

            # [v0.4] Intermediate FID uses fast DPM-Solver; final FID uses configured sampler
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
            log.info(f"FID @ epoch {epoch} ({tag}, {sampler_cfg.sampler} {sampler_cfg.sample_steps} steps): "
                     f"{fid:.2f}  [{time.time()-t_fid_start:.0f}s]")

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

    # ──  final gallery (for the report) ───────────────────
    log.info(f"Generating final gallery of {cfg.final_gallery_n} images...")
    gen = torch.Generator(device=device).manual_seed(MASTER_SEED + 1)
    gallery = []
    remaining = cfg.final_gallery_n
    while remaining > 0:
        batch_n = min(cfg.n_samples, remaining)
        tmp_cfg = Config(**{**asdict(cfg), "n_samples": batch_n})
        imgs = sample(ema.shadow, sched, tmp_cfg, device, generator=gen)
        gallery.append(imgs)
        remaining -= batch_n
    gallery = torch.cat(gallery, dim=0)[:cfg.final_gallery_n]
    gallery_grid = make_grid(gallery * 0.5 + 0.5, nrow=10)
    save_image(gallery_grid, img_dir / "final_gallery.png")
    log.info(f"Saved final gallery → final_gallery.png")

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

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Training complete.")
    return history


# ──────────────────────────────────────────────────────────────
# 10. PLOTTING
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

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
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
# 11. ENTRY POINT
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDPM pipeline v0.4 — trained on REAL faces")
    p.add_argument("--phase",   default="1",
                   help="1 | 2 | 3 | 4 | all | range like 2-4 | list like 2,3,4")
    p.add_argument("--epochs",  type=int, default=None, help="Override epochs")
    p.add_argument("--batch",   type=int, default=None, help="Override batch size")
    p.add_argument("--log_dir", default="./ddpm_runs_v04", help="Output directory")
    p.add_argument("--warmup",  type=int, default=None, help="Override warmup steps")
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
    if args.epochs is not None: overrides["epochs"]       = args.epochs
    if args.batch  is not None: overrides["batch_size"]   = args.batch
    if args.warmup is not None: overrides["warmup_steps"] = args.warmup
    if args.no_fp16:             overrides["use_fp16"]    = False

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
        print(f"\n{'='*60}")
        print(f"  Phase {ph}: {cfg.phase_name}  [v0.4]")
        print(f"  attention={cfg.use_attention}  schedule={cfg.schedule}  sampler={cfg.sampler}({cfg.sample_steps})")
        print(f"  optimizer=AdamW(wd={cfg.weight_decay})  lr={cfg.lr}→{cfg.lr_min}")
        print(f"  warmup={cfg.warmup_steps} steps → cosine  |  master_seed={MASTER_SEED}")
        print(f"  FID every {cfg.fid_every} epochs (fast: DPM-Solver 20 steps); final FID uses configured sampler")
        print(f"{'='*60}\n")
        h = train(cfg, device)
        all_histories.append(h)

    if len(all_histories) > 1:
        plot_results(all_histories, args.log_dir)

    print("\n" + "=" * 60)
    print(f"{'Phase':<30}  {'Final FID':>10}  {'1000-step(s)':>12}  {'20-step(s)':>10}")
    print("-" * 60)
    for h in all_histories:
        fid_str = f"{h['fid_final']:.2f}" if h["fid_final"] is not None else "n/a"
        t1000   = f"{h['sample_time_1000']:.1f}" if h["sample_time_1000"] else "n/a"
        t20     = f"{h['sample_time_fast']:.1f}"  if h["sample_time_fast"]  else "n/a"
        print(f"{h['phase']:<30}  {fid_str:>10}  {t1000:>12}  {t20:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()