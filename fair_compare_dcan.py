"""
Fair DCGAN vs DDPM Comparison
=============================
Re-evaluates a trained DCGAN generator with the SAME FID protocol used for
the DDPM models (v0.6 / v0.7):
  - same number of generated samples (2048 by default)
  - same real reference images (held-out val set from the DDPM pipeline)
  - same compute_fid function from the v0.7 pipeline

The DCGAN script does NOT save checkpoints by default — see the patch below
that adds `torch.save(...)` at the end of training. After re-training once with
that patch, point CKPT_PATH below to the saved generator weights.
"""

import sys, torch, time, json
from pathlib import Path
from torchvision.utils import save_image, make_grid

# ----- Load v0.7 pipeline (for compute_fid + dataloaders, same as before) ----
import importlib.util
spec = importlib.util.spec_from_file_location("v07", "/home/up202512956/ddpm_pipeline_v0.7.py")
v07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v07)

compute_fid       = v07.compute_fid
build_dataloaders = v07.build_dataloaders
Config            = v07.Config
MASTER_SEED       = v07.MASTER_SEED

# ----- Load DCGAN Generator class ---------------------------------------------
# Adjust this path to wherever your DCGAN script lives on the cluster
DCGAN_SCRIPT = "/home/up202512956/dcgan_upgraded.py"
spec2 = importlib.util.spec_from_file_location("dcgan", DCGAN_SCRIPT)
dcgan = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(dcgan)
Generator = dcgan.Generator
LATENT_DIM = dcgan.LATENT_DIM
IMAGE_SIZE = dcgan.IMAGE_SIZE   # 64

# ----- Config -----------------------------------------------------------------
CKPT_PATH      = "/data/01/up202512956/dcgan_runs/upgrade_epochs_10/generator_final.pt"
N_FID_SAMPLES  = 2048
BATCH_SIZE     = 64
RESULT_JSON    = "/data/01/up202512956/fair_comparison_dcgan.json"
SAMPLES_PNG    = "/data/01/up202512956/dcgan_fid_samples.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load trained DCGAN generator
    print(f"Loading DCGAN generator: {CKPT_PATH}")
    generator = Generator(latent_dim=LATENT_DIM).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    generator.load_state_dict(state)
    generator.eval()
    print(f"DCGAN generator loaded  (latent_dim={LATENT_DIM}, image_size={IMAGE_SIZE})")

    # 2) Build the SAME real reference images as the DDPM pipeline uses
    print("\nBuilding DDPM pipeline dataloaders for real reference images...")
    cfg = Config()                  # default DDPM config
    cfg.image_size    = IMAGE_SIZE  # match DCGAN's 64x64 (was 128 in v0.7)
    cfg.fid_n_samples = N_FID_SAMPLES
    _, val_loader, _ = build_dataloaders(cfg)

    real_list = []
    for xr in val_loader:
        real_list.append(xr)
        if sum(r.size(0) for r in real_list) >= N_FID_SAMPLES:
            break
    real_fid_imgs = torch.cat(real_list, dim=0)[:N_FID_SAMPLES]
    print(f"FID reference: {real_fid_imgs.size(0)} real images @ {IMAGE_SIZE}x{IMAGE_SIZE}")

    # 3) Generate fake samples from DCGAN
    print(f"\nGenerating {N_FID_SAMPLES} samples from DCGAN...")
    fake_list = []
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(MASTER_SEED + 100)

    with torch.no_grad():
        while sum(f.size(0) for f in fake_list) < N_FID_SAMPLES:
            z = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=device, generator=gen)
            imgs = generator(z)        # output in [-1, 1] (Tanh)
            fake_list.append(imgs.cpu())
            n_done = sum(f.size(0) for f in fake_list)
            print(f"  {n_done}/{N_FID_SAMPLES} samples")

    fake_fid_imgs = torch.cat(fake_list, dim=0)[:N_FID_SAMPLES]
    sample_time = time.time() - t0
    print(f"Generation complete: {sample_time:.1f}s "
          f"({sample_time/N_FID_SAMPLES*1000:.1f} ms per image)")

    # 4) Compute FID with the SAME function used for DDPM
    print("\nComputing FID (same compute_fid as v0.6/v0.7)...")
    fid = compute_fid(real_fid_imgs, fake_fid_imgs, device)

    # 5) Save a 4x4 sample grid for visual inspection
    grid = make_grid(fake_fid_imgs[:16] * 0.5 + 0.5, nrow=4)
    save_image(grid, SAMPLES_PNG)
    print(f"Saved sample grid → {SAMPLES_PNG}")

    # 6) Report
    print()
    print("=" * 70)
    print("FAIR COMPARISON RESULT — DCGAN vs DDPM")
    print("=" * 70)
    print(f"v0.6 Phase 3 (DDPM 1000):  FID = 66.20   [DDPM baseline]")
    print(f"v0.7 Phase 4 (DDPM 1000):  FID = 59.53   [DDPM best]")
    print(f"DCGAN (upgraded, 10 ep):   FID = {fid:.2f}   [GAN baseline]")
    print("=" * 70)

    result = {
        "v06_phase3_ddpm1000":   66.20,
        "v07_phase4_ddpm1000":   59.53,
        "dcgan_upgraded":        float(fid),
        "n_samples":             N_FID_SAMPLES,
        "image_size":            IMAGE_SIZE,
        "sample_time_seconds":   sample_time,
        "ckpt_path":             CKPT_PATH,
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result → {RESULT_JSON}")


if __name__ == "__main__":
    main()