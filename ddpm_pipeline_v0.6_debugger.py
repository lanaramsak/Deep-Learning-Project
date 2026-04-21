"""
Sampler Debug — v0.6
=====================
Loads a trained DDPM checkpoint and tests multiple samplers side-by-side
to diagnose Phase 4's DPM-Solver failure in v0.6.

What this script does:
  1. Load v0.6 Phase 3 (shifted_cosine) EMA weights from checkpoint
  2. Generate a 4×4 grid with each of these samplers:
       - DDPM (1000 steps)     [reference: known to work]
       - DDIM (50 steps)       [deterministic, moderate speedup]
       - DPM-Solver++ (20)     [v0.6's broken config]
       - DPM-Solver++ (50)     [v0.6's still-broken config]
       - DPM-Solver++ (100)    [can more steps rescue it?]
  3. Save each grid. If DDIM works but DPM-Solver doesn't → it's specifically
     a DPM-Solver+v-prediction interaction bug (not a general sampler issue).

Usage on HPC:
  sbatch -p normal --qos gpu_batch --gres=gpu:1 --time=2:00:00 \
    --job-name "dbg_samp" \
    --wrap "python3 ~/sampler_debug.py \
      --ckpt /data/01/up202512956/ddpm_runs_v06/phase3_shifted_cosine/checkpoints/ckpt_epoch_0080.pt \
      --out  /data/01/up202512956/sampler_debug"
"""

import argparse, copy, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

# Import v0.6 components directly — the filename has a dot (ddpm_pipeline_v0.6.py)
# which Python cannot import normally, so we load it via importlib.
import importlib.util
_v06_path = Path(__file__).parent / "ddpm_pipeline_v0.6.py"
if not _v06_path.exists():
    raise FileNotFoundError(f"Expected v0.6 pipeline at {_v06_path}")
_spec = importlib.util.spec_from_file_location("ddpm_v06", str(_v06_path))
_v06  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v06)

UNet                 = _v06.UNet
NoiseScheduler       = _v06.NoiseScheduler
Config               = _v06.Config
ddpm_sample          = _v06.ddpm_sample
ddim_sample          = _v06.ddim_sample
dpm_solver_pp_sample = _v06.dpm_solver_pp_sample
MASTER_SEED          = _v06.MASTER_SEED


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load a v0.6 checkpoint and reconstruct model + scheduler."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg_dict = ckpt["config"]
    print(f"Checkpoint config: phase={cfg_dict['phase_name']}  "
          f"prediction={cfg_dict.get('prediction_type', '?')}  "
          f"schedule={cfg_dict['schedule']}  epoch={ckpt['epoch']}")

    # Rebuild model with exact same config
    model = UNet(
        channels=cfg_dict["channels"],
        use_attention=cfg_dict["use_attention"],
        attn_resolutions=cfg_dict["attn_resolutions"],
        image_size=cfg_dict["image_size"],
        dropout=cfg_dict["dropout"],
        use_checkpoint=False,
    ).to(device)

    # Load EMA weights (used for sampling)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    # Rebuild noise scheduler
    sched = NoiseScheduler(
        T=cfg_dict["T"],
        schedule=cfg_dict["schedule"],
        beta_start=cfg_dict["beta_start"],
        beta_end=cfg_dict["beta_end"],
        cosine_shift=cfg_dict.get("cosine_shift", 2.0),
    ).to(device)

    # Reconstruct Config object for sample dispatch
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})

    return model, sched, cfg


def run_sampler_test(model, sched, cfg, device, sampler_name: str, steps: int,
                     n_samples: int = 16, seed: int = MASTER_SEED):
    """Run one sampler and return images + time."""
    shape = (n_samples, 3, cfg.image_size, cfg.image_size)
    gen = torch.Generator(device=device).manual_seed(seed)
    pt = cfg.prediction_type

    t0 = time.time()
    if sampler_name == "ddpm":
        imgs = ddpm_sample(model, sched, shape, device, steps, gen, pt)
    elif sampler_name == "ddim":
        imgs = ddim_sample(model, sched, shape, device, steps, 0.0, gen, pt)
    elif sampler_name == "dpm_solver":
        imgs = dpm_solver_pp_sample(model, sched, shape, device, steps, gen, pt)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    elapsed = time.time() - t0
    return imgs, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to v0.6 checkpoint .pt file")
    p.add_argument("--out",  default="./sampler_debug", help="Output directory")
    p.add_argument("--n",    type=int, default=16, help="Samples per grid")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, sched, cfg = load_checkpoint(args.ckpt, device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Prediction type: {cfg.prediction_type}")
    print()

    # Test matrix — sampler × step count
    tests = [
        ("ddpm",        1000),   # reference (known good)
        ("ddim",          50),   # alt 50-step deterministic
        ("ddim",         100),   # alt 100-step
        ("dpm_solver",    20),   # v0.5 config (failed)
        ("dpm_solver",    50),   # v0.6 config (still failed)
        ("dpm_solver",   100),   # more steps
        ("dpm_solver",   200),   # even more
    ]

    results = []
    for sampler_name, steps in tests:
        print(f"\n→ {sampler_name} @ {steps} steps ...")
        try:
            imgs, elapsed = run_sampler_test(model, sched, cfg, device,
                                              sampler_name, steps, args.n)
            tag = f"{sampler_name}_{steps:04d}"
            out_path = out_dir / f"{tag}.png"
            grid = make_grid(imgs * 0.5 + 0.5, nrow=int(args.n ** 0.5))
            save_image(grid, out_path)

            # Quick sanity check — mean/std of pixel values
            mean = imgs.mean().item()
            std = imgs.std().item()
            print(f"   saved → {out_path.name}   [{elapsed:.1f}s]   mean={mean:+.3f}  std={std:.3f}")
            results.append((tag, elapsed, mean, std))
        except Exception as e:
            print(f"   FAILED: {e}")
            results.append((f"{sampler_name}_{steps:04d}", -1, 0, 0))

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Sampler':<22}  {'Time(s)':>8}  {'Mean':>7}  {'Std':>6}")
    print("-" * 60)
    for tag, t, m, s in results:
        t_str = f"{t:.1f}" if t > 0 else "FAIL"
        print(f"{tag:<22}  {t_str:>8}  {m:+.3f}  {s:.3f}")
    print("=" * 60)
    print("\nInterpretation hints:")
    print("  - std ≈ 0.5 and mean ≈ 0 → good (natural image statistics)")
    print("  - std > 0.8 or mean far from 0 → likely noise/broken output")
    print("  - If DDIM works but DPM-Solver doesn't → DPM+v-pred interaction bug")


if __name__ == "__main__":
    main()