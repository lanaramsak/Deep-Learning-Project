"""
DDPM Face Generator — trainiert auf echten Gesichtern (wiki-Datensatz)
=======================================================================
Ziel: Das Modell lernt die Verteilung echter Gesichter und generiert
      danach neue, synthetische Gesichtsbilder.

Aktuell (low-compute Modus):
  - 500 Trainingsbilder, 64×64 px, 30 Epochen
  - Generiert 10 neue Bilder nach dem Training

Zum Hochskalieren (bessere Qualität):
  - Alle Kommentare mit [SCALE UP] zeigen die relevanten Stellen

Ausführen:
  python Image-generator_v0.1.py
"""

import math
import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════
#  KONFIGURATION — hier alle wichtigen Parameter auf einen Blick
# ══════════════════════════════════════════════════════════════════

# Pfad zum wiki-Ordner mit echten Gesichtern (eine Ebene über dem Projekt)
REAL_IMAGE_DIR = Path(__file__).parent.parent / "wiki"

# Ausgabeordner für generierte Bilder
OUTPUT_DIR = Path(__file__).parent / "generated"

# Trainingsparameter
N_TRAIN_IMAGES = 2000        # [SCALE UP] None → alle verfügbaren Bilder verwenden
IMAGE_SIZE     = 64         # [SCALE UP] 128 oder 256 für deutlich bessere Gesichtsqualität
EPOCHS         = 25          # CPU: 5 Epochen (~15 Min) | [SCALE UP] GPU: 100+
BATCH_SIZE     = 16         # CPU: 16 | [SCALE UP] GPU: 64
LR             = 2e-4       # Lernrate (gut bewährt für DDPM)

# Diffusionsparameter
T              = 1000       # Anzahl Diffusionsschritte (nicht ändern)
SCHEDULE       = "cosine"   # "cosine" ist besser als "linear" (Nichol & Dhariwal 2021)

# U-Net Kanäle: bestimmt Modellgröße und Qualität
# CPU: [32, 64, 128] → ~500K Parameter, ~3 Min/Epoche
# [SCALE UP] GPU: [64, 128, 256] oder [128, 256, 512]
# CHANNELS       = [128, 256, 512]
# CHANNELS       = [64, 128, 256]
CHANNELS       = [32, 64, 128]

# Sampling: DPM-Solver++ braucht nur 20 Schritte statt 1000 — viel schneller
SAMPLE_STEPS   = 20
N_GENERATE     = 10         # Anzahl zu generierender Bilder

SEED           = 42


# ══════════════════════════════════════════════════════════════════
#  DATENSATZ
# ══════════════════════════════════════════════════════════════════

class RealFaceDataset(Dataset):
    """Lädt echte Gesichtsbilder aus dem wiki-Ordner."""

    def __init__(self, image_dir: Path, n_samples: int = None, image_size: int = 64):
        # Alle Bild-Dateien einsammeln
        all_paths = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png"))

        if len(all_paths) == 0:
            raise FileNotFoundError(f"Keine Bilder gefunden in: {image_dir}")

        # Optional: nur eine Teilmenge verwenden (low-compute Modus)
        if n_samples is not None and n_samples < len(all_paths):
            rng = random.Random(SEED)
            all_paths = rng.sample(all_paths, n_samples)

        self.paths = all_paths
        print(f"Datensatz: {len(self.paths)} Bilder aus {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Color Jitter: Helps the model handle different lighting conditions
            # brightness/contrast/saturation: 0.1 is subtle but effective
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # transforms.RandomRotation(degrees=5), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ══════════════════════════════════════════════════════════════════
#  NOISE SCHEDULER
#  Verwaltet den Rausch-Zeitplan: wie viel Rauschen bei welchem Schritt
# ══════════════════════════════════════════════════════════════════

class NoiseScheduler:
    """
    Implementiert den DDPM-Rauschplan.

    Forward-Prozess (Training):
      q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)·x_0, (1−ᾱ_t)·I)
      → Bild x_0 wird schrittweise verrauscht bis zu reinem Rauschen x_T

    Reverse-Prozess (Generierung):
      p(x_{t-1} | x_t) — das U-Net lernt, das Rauschen vorherzusagen
    """

    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        self.T = T

        if schedule == "cosine":
            # Cosine-Schedule: bessere Qualität als linear, v.a. am Anfang/Ende
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = (1.0 - alphas_bar[1:] / alphas_bar[:-1]).float().clamp(1e-4, 0.9999)
        else:  # linear
            betas = torch.linspace(1e-4, 2e-2, T, dtype=torch.float32)

        alphas      = 1.0 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        # Vorberechnete Werte für schnelles Training
        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bar  = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_ab    = alpha_bar.sqrt()
        self.sqrt_1mab  = (1 - alpha_bar).sqrt()

        # Posterior für DDPM-Sampling
        self.post_var   = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
        self.post_log_var = self.post_var.clamp(min=1e-20).log()
        self.post_c1    = betas * alpha_bar_prev.sqrt() / (1 - alpha_bar)
        self.post_c2    = (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar)

    def to(self, device):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def add_noise(self, x0, t, noise=None):
        """Forward-Prozess: fügt Rauschen in einem Schritt hinzu (geschlossene Form)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s  = self.sqrt_ab[t][:, None, None, None]
        s1 = self.sqrt_1mab[t][:, None, None, None]
        return s * x0 + s1 * noise, noise


# ══════════════════════════════════════════════════════════════════
#  U-NET ARCHITEKTUR
#  Das neuronale Netz, das bei jedem Timestep das Rauschen vorhersagt
# ══════════════════════════════════════════════════════════════════

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Wandelt den Zeitschritt t in einen Vektor um (wie Positional Encoding)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    """
    Residual-Block mit Zeitschritt-Konditionierung.
    Das Netz weiß immer bei welchem Rauschgrad (t) es arbeitet.
    """
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1  = nn.GroupNorm(min(32, in_ch // 4), in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)     # Zeitschritt einspeisen
        self.norm2  = nn.GroupNorm(min(32, out_ch // 4), out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Self-Attention im Bottleneck: lässt das Netz globale Bildstruktur lernen.
    Besonders wichtig für Gesichter (Symmetrie, Proportionen).
    """
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=8, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class UNet(nn.Module):
    """
    U-Net Denoiser für IMAGE_SIZE × IMAGE_SIZE Bilder.

    Encoder:    64 → 32 → 16  (Downsampling)
    Bottleneck: 16×16         (Self-Attention hier — lernt globale Struktur)
    Decoder:    16 → 32 → 64  (Upsampling mit Skip-Connections)

    [SCALE UP] channels=[128, 256, 512] für mehr Modellkapazität
    """
    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = CHANNELS
        C0, C1, C2 = channels
        # t_dim = C0 * 4   # Dimension der Zeitschritt-Embeddings

        TEMB_DIM = 128  # fixed, independent of channel size
        t_dim = TEMB_DIM * 4  # = 512
        self.time_mlp = nn.Sequential(
            nn.Linear(TEMB_DIM, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        # Encoder
        self.in_conv = nn.Conv2d(3, C0, 3, padding=1)
        self.down1_res1 = ResBlock(C0, C1, t_dim)
        self.down1_res2 = ResBlock(C1, C1, t_dim)
        self.down1_pool = nn.Conv2d(C1, C1, 3, stride=2, padding=1)   # 64 → 32

        self.down1_attn = SelfAttention(C1)

        self.down2_res1 = ResBlock(C1, C2, t_dim)
        self.down2_res2 = ResBlock(C2, C2, t_dim)
        self.down2_pool = nn.Conv2d(C2, C2, 3, stride=2, padding=1)   # 32 → 16

        # Bottleneck mit Self-Attention (lernt globale Gesichtsstruktur)
        self.mid1 = ResBlock(C2, C2, t_dim)
        self.attn = SelfAttention(C2)
        self.mid2 = ResBlock(C2, C2, t_dim)

        # Decoder (mit Skip-Connections vom Encoder)
        self.up1_up   = nn.ConvTranspose2d(C2, C2, 2, stride=2)       # 16 → 32
        self.up1_res1 = ResBlock(C2 + C2, C1, t_dim)
        self.up1_res2 = ResBlock(C1, C1, t_dim)

        self.up1_attn = SelfAttention(C1)

        self.up2_up   = nn.ConvTranspose2d(C1, C1, 2, stride=2)       # 32 → 64
        self.up2_res1 = ResBlock(C1 + C1, C0, t_dim)
        self.up2_res2 = ResBlock(C0, C0, t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(32, C0),
            nn.SiLU(),
            nn.Conv2d(C0, 3, 3, padding=1),
        )

        # Gewichte initialisieren
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        # t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))
        TEMB_DIM = 128
        t_emb = self.time_mlp(sinusoidal_embedding(t, TEMB_DIM))

        # Encoder
        x  = self.in_conv(x)
        x  = self.down1_res2(self.down1_res1(x, t_emb), t_emb)
        x = self.down1_attn(x)
        s1 = x                                              # Skip-Connection speichern
        x  = self.down1_pool(x)

        x  = self.down2_res2(self.down2_res1(x, t_emb), t_emb)
        s2 = x
        x  = self.down2_pool(x)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.attn(x)
        x = self.mid2(x, t_emb)

        # Decoder
        x = torch.cat([self.up1_up(x), s2], dim=1)
        x = self.up1_res2(self.up1_res1(x, t_emb), t_emb)
        x = self.up1_attn(x)

        x = torch.cat([self.up2_up(x), s1], dim=1)
        x = self.up2_res2(self.up2_res1(x, t_emb), t_emb)

        return self.out(x)


# ══════════════════════════════════════════════════════════════════
#  EMA (Exponential Moving Average)
#  Glättet die Modellgewichte über die Zeit → stabilere Generierung
# ══════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)


# ══════════════════════════════════════════════════════════════════
#  SAMPLING — DPM-Solver++ (Lu et al. 2022)
#  Generiert neue Bilder in nur 20 Schritten statt 1000
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_images(model: nn.Module, sched: NoiseScheduler,
                    n: int, device: torch.device, steps: int = 20) -> torch.Tensor:
    """
    DPM-Solver++(2M): schneller und hochwertiger als Standard-DDPM-Sampling.
    Braucht nur 20 statt 1000 Schritte ohne Qualitätsverlust.
    """
    model.eval()
    shape = (n, 3, IMAGE_SIZE, IMAGE_SIZE)

    ac    = sched.alpha_bar.to(device)
    alpha = ac.sqrt()
    sigma = (1 - ac).sqrt()
    lam   = torch.log(alpha / sigma)   # λ-Raum für DPM-Solver

    # Zeitschritte gleichmäßig von T-1 bis 0
    seq = torch.linspace(sched.T - 1, 0, steps + 1, dtype=torch.long, device=device)

    x = torch.randn(shape, device=device)   # Start: reines Rauschen
    D_prev, h_prev = None, None

    for i in tqdm(range(steps), desc="Generiere Bilder", leave=False):
        t_s = seq[i].item()
        t_t = seq[i + 1].item()
        t_batch = torch.full((n,), t_s, device=device, dtype=torch.long)

        # Rausch-Vorhersage des U-Net → x0-Schätzung
        eps    = model(x, t_batch)
        D_curr = ((x - sigma[t_s] * eps) / alpha[t_s]).clamp(-1, 1)

        h = (lam[t_t] - lam[t_s]).item()

        # 2nd-order Update: nutzt vorherigen Schritt für genauere Annäherung
        D_eff = D_curr
        if D_prev is not None:
            r     = h_prev / h
            D_eff = (1 + 0.5 / r) * D_curr - (0.5 / r) * D_prev

        coeff_x = alpha[t_t] / alpha[t_s]
        coeff_d = sigma[t_t] * (-torch.expm1(torch.tensor(-h)))
        x = coeff_x * x + coeff_d * D_eff

        D_prev, h_prev = D_curr, h

    return x.clamp(-1, 1)


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════

def train(model, ema, sched, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining gestartet: {EPOCHS} Epochen, {len(loader.dataset)} Bilder")
    print(f"Modell: {sum(p.numel() for p in model.parameters()):,} Parameter\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x0 in tqdm(loader, desc=f"Epoche {epoch:02d}/{EPOCHS}", leave=False):
            x0 = x0.to(device)
            B  = x0.size(0)

            # Zufällige Zeitschritte für jedes Bild im Batch
            t = torch.randint(0, T, (B,), device=device)

            # Forward-Prozess: Rauschen hinzufügen
            xt, noise = sched.add_noise(x0, t)

            # U-Net sagt das Rauschen vorher — das ist der Trainingsverlust
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoche {epoch:02d}/{EPOCHS} | Verlust: {avg_loss:.4f}")

    return model, ema


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerät: {device}")
    if device.type == "cpu":
        print("[Hinweis] Kein GPU gefunden — Training auf CPU ist langsam.\n")

    # 1. Datensatz laden
    dataset = RealFaceDataset(REAL_IMAGE_DIR, n_samples=N_TRAIN_IMAGES, image_size=IMAGE_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, drop_last=True)

    # 2. Modell und Scheduler initialisieren
    sched = NoiseScheduler(T=T, schedule=SCHEDULE).to(device)
    model = UNet(channels=CHANNELS).to(device)
    ema   = EMA(model)

    # 3. Training
    model, ema = train(model, ema, sched, loader, device)

    # 4. Bilder generieren (EMA-Modell für stabilere Ergebnisse)
    print(f"\nGeneriere {N_GENERATE} neue Gesichtsbilder...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = generate_images(ema.shadow, sched, n=N_GENERATE, device=device, steps=SAMPLE_STEPS)

    # Bilder von [-1, 1] → [0, 1] und einzeln speichern
    imgs_01 = generated * 0.5 + 0.5
    for i, img in enumerate(imgs_01):
        save_image(img, OUTPUT_DIR / f"generated_{i+1:02d}.png")

    # Zusätzlich ein Grid-Bild mit allen generierten Bildern
    grid = make_grid(imgs_01, nrow=5)
    save_image(grid, OUTPUT_DIR / "grid.png")

    print(f"\nFertig! {N_GENERATE} Bilder gespeichert in: {OUTPUT_DIR}/")
    print(f"  → Einzelbilder: generated_01.png bis generated_{N_GENERATE:02d}.png")
    print(f"  → Übersicht:    grid.png")
    print(f"\nHinweise zum Hochskalieren:")
    print(f"  - N_TRAIN_IMAGES = None    → alle verfügbaren Bilder nutzen")
    print(f"  - IMAGE_SIZE = 128         → deutlich bessere Gesichtsqualität")
    print(f"  - EPOCHS = 100             → mehr Training = bessere Qualität")
    print(f"  - CHANNELS = [128,256,512] → größeres Modell")


if __name__ == "__main__":
    main()
