#!/bin/bash
#SBATCH --job-name=0%-Retina-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=gpu-l40s                  # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores
#SBATCH --mem-per-gpu=48G                     # Memory per GPU
#SBATCH --gres=gpu:1                          # Request 4 GPUs

python train.py
