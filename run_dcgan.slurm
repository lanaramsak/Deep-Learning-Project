#!/bin/bash
#SBATCH --job-name=multiple_models
#SBATCH --output=multiple_models.%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G

cd /home/up202512959/Deep-Learning-Project
source .venv/bin/activate

python -m pip install nbconvert jupyter
python -m nbconvert --to notebook --execute --inplace multiple_models.ipynb
