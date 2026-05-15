#!/bin/bash
#SBATCH --job-name=multiple_models
#SBATCH --output=multiple_models.%j.out
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=16G

cd /home/up202512959/Deep-Learning-Project
source .venv/bin/activate

python -m nbconvert --to notebook --execute --inplace multiple_models2.ipynb
python dcgan_generator_upgraded.py --epochs 100 --run-name fid_test --instance-noise-std 0.1 --fid-samples 256


