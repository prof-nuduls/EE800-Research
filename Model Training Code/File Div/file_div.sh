#!/bin/bash
#SBATCH --job-name=Train_div
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=10:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G

module load python39

srun python Train_div.py

