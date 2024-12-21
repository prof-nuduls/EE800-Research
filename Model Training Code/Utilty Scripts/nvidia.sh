#!/bin/bash
#SBATCH --job-name=nvidia      # Job name
#SBATCH --output=update_out_%j.txt          # Standard output log
#SBATCH --error=update_err_%j.txt            # Standard error log
#SBATCH --time=0:01:00              # Time limit (10 hours)
#SBATCH --partition=gpu-l40s        # GPU partition
#SBATCH --ntasks=1                      # One task
#SBATCH --gres=gpu:1              # Request 4 CPU cores
#SBATCH --mem=1G                      # Memory request (150 GB)


# Run the script using all 4 GPUs (0,1,2,3)
srun nvidia-smi
