#!/bin/bash
# ============================================================
# Ablation Study — v0.6 Training Dynamics Decomposition
# ============================================================
# Takes the best v0.6 config (Phase 3: shifted_cosine + v-pred + Min-SNR)
# and disables ONE feature at a time to measure its individual contribution.
#
# Variants:
#   A1: no v-prediction (use eps-prediction instead)       → tests v-pred value
#   A2: no Min-SNR weighting (uniform weighting instead)   → tests Min-SNR value
#   A3: no shifted cosine (standard cosine instead)        → tests shift value
#
# The baseline to compare against is the existing v06 phase3 run (FID 66.20).
# Each variant runs 80 epochs at 128×128. Expected runtime: 6-8h each.
#
# Usage:
#   bash ~/ablation_v06.sh
# ============================================================

LOG=/data/01/up202512956/ddpm_runs_v06_ablation
mkdir -p "$LOG"

# --- A1: eps-prediction instead of v-prediction ---
sbatch -p normal --qos gpu_batch --gres=gpu:1 --time=16:00:00 \
  --job-name "abl_no_vpred" \
  --wrap "python3 ~/ddpm_pipeline_v0.6.py --phase 3 --epochs 80 \
          --prediction eps \
          --log_dir $LOG/no_vpred"

# --- A2: uniform loss weighting instead of Min-SNR ---
sbatch -p normal --qos gpu_batch --gres=gpu:1 --time=16:00:00 \
  --job-name "abl_no_minsnr" \
  --wrap "python3 ~/ddpm_pipeline_v0.6.py --phase 3 --epochs 80 \
          --weighting uniform \
          --log_dir $LOG/no_minsnr"

# --- A3: standard cosine instead of shifted cosine ---
# No CLI flag for schedule type, so we override via env var by editing the
# Config default. Easier: submit Phase 2 (which uses linear schedule) with
# schedule patched — but cleanest is a small wrapper that overrides.
# For simplicity: run Phase 3 but disable the shift by setting cosine_shift=1.0
# via a one-line sed patch before running.

# Create a patched copy of the pipeline with cosine_shift=1.0 (= standard cosine)
cp ~/ddpm_pipeline_v0.6.py ~/ddpm_pipeline_v0.6_stdcos.py
sed -i 's/cosine_shift: float     = 2.0/cosine_shift: float     = 1.0/' ~/ddpm_pipeline_v0.6_stdcos.py

sbatch -p normal --qos gpu_batch --gres=gpu:1 --time=16:00:00 \
  --job-name "abl_no_shift" \
  --wrap "python3 ~/ddpm_pipeline_v0.6_stdcos.py --phase 3 --epochs 80 \
          --log_dir $LOG/no_shift"

squeue -u up202512956
echo
echo "Baseline for comparison: v06 phase3_shifted_cosine FID = 66.20"
echo "Results will be in: $LOG/{no_vpred,no_minsnr,no_shift}/phase3_shifted_cosine/history.json"