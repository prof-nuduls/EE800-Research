import os

# Base directory path where Train_RCNN.sh files should be created
base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/Faster_RCNN/model/models/300_epochs_full"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Template for the Train_RCNN.sh file
template = """#!/bin/bash
#SBATCH --job-name={folder}-FR-300epochs      # Job name reflecting folder/model
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

python -m torch.distributed.launch --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --use-env train.py  \
    --data "/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/data_configs/data.yaml" \
    --epochs 300 \
    --model fasterrcnn_resnet50_fpn_v2 \
    --name {folder}_Faster_RCNN_v2_p30 \
    --batch 8 \
    --imgsz 1920 \
    --patience 50
"""

# Loop through each folder and write the Train_RCNN.sh file
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "Train_RCNN_2.sh")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Write the Train_RCNN.sh file with the correct folder name and settings
    with open(file_path, 'w') as file:
        file.write(template.format(folder=folder))
    
    print(f"Created {file_path}")
