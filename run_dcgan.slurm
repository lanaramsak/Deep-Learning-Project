#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --output=vit.%j.out
#SBATCH --error=vit.%j.err
#SBATCH --time=12:00:00
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
# python dcgan_generator_upgraded.py --epochs 50
# python dcgan_generator_upgraded.py --epochs 50 --run-name hinge_noise --instance-noise-std 0.1
# python dcgan_generator_upgraded.py --epochs 100
# python dcgan_generator_upgraded.py --epochs 50 --run-name hinge_noise --instance-noise-std 0.05
# python dcgan_generator.py --epochs 5
# python dcgan_generator.py --epochs 20
# python dcgan_generator.py --epochs 50
# python two_branch_ablation.py
# python two_branch_combo_experiment.py --best-metric val_auc
# python ensemble_with_two_branch.py --two-branch-checkpoint results/two_branch_combo/rotation_15_blur_2.0_best.pt --rotation-degrees 15 --blur-radius 2.0

python VisionTransformer.py --n 500 --batch-size 32 --epochs 10
