#!/bin/bash
#SBATCH --job-name=+90kTest-Retina-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=compute                 # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores

python inference.py --weights /mmfs1/home/dmiller10/EE800\ Research/Code/RetinaNet/+90k/runs/training/retinanet/best_model.pth --input /mmfs1/home/dmiller10/EE800\ Research/Data/SeaDronesSee\ Object\ Detection\ v2/Uncompressed\ Version/Test/images --no-labels --imgsz 640  