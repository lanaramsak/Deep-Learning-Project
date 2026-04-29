#!/bin/bash
#SBATCH --job-name=dcgan
#SBATCH --output=dcgan.%j.out
#SBATCH --error=dcgan.%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Če moraš naloadati modul za Python, odkomentiraj eno od spodnjih vrstic:
# module load python/3.9
# module load cuda/12.1

cd /home/up202512959/Deep-Learning-Project
source .venv/bin/activate
python Image_Generation/dcgan_generator.py --epochs 5
python Image_Generation/dcgan_generator.py --epochs 20
python Image_Generation/dcgan_generator.py --epochs 50
python two_branch_ablation.py
