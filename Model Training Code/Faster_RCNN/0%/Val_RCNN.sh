#!/bin/bash
#SBATCH --job-name=0%-FR-val     # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=compute                 # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores

python eval.py --model fasterrcnn_resnet50_fpn_v2 --weights /outputs/training/0%_Faster_RCNN_v2_p30/best_model.pth --data "/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/0%/data_configs/data.yaml" --imgsz 1920 --batch 4 --verbose