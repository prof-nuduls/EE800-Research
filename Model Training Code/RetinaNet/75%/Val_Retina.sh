#!/bin/bash
#SBATCH --job-name=75%Val-Retina-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=compute                 # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores

python eval.py --weights /mmfs1/home/dmiller10/EE800 Research/Code/RetinaNet/75%/runs/training/retinanet/best_model.pth --input-images /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/75%/Valid/images --input-annots /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/75%/Valid/annotations
