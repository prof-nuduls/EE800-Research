#!/bin/bash
#SBATCH --job-name=+90k-FR-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=gpu-l40s                  # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores
#SBATCH --mem-per-gpu=48G                     # Memory per GPU
#SBATCH --gres=gpu:1                        # Request 4 GPUs

export MASTER_ADDR=localhost
export MASTER_PORT=29501
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py  \
    --data "/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/+90k/data_configs/data.yaml" \
    --epochs 300 \
    --resume-training \
    --weights ./outputs/training/+90k_Faster_RCNN_v2_p30/last_model.pth \
    --model fasterrcnn_resnet50_fpn_v2 \
    --name +90k_Faster_RCNN_v2_p30 \
    --batch 8 \
    --imgsz 1920 \
    --patience 50 \
    --disable-wandb
