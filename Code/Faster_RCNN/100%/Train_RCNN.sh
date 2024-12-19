#!/bin/bash
#SBATCH --job-name=0%-FR-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=gpu-h100                  # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores
#SBATCH --mem-per-gpu=48G                     # Memory per GPU
#SBATCH --gres=gpu:2                         # Request 4 GPUs

export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
    --data "/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/0%/data_configs/data.yaml" \
    --epochs 300 \
    --model fasterrcnn_resnet50_fpn_v2 \
    --name 0%_Faster_RCNN_v2_p30 \
    --batch 16 \
    --disable-wandb \
    --imgsz 1920 \
    --patience 10
