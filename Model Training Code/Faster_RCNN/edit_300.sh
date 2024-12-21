#!/bin/bash

# Base directory where folders like 0%, 25%, etc., are located
BASE_DIR="/mmfs1/home/dmiller10/EE800 Research/Code/Faster_RCNN/model/models/300_epochs"

# Template for the script
SCRIPT_TEMPLATE="#!/bin/bash
#SBATCH --job-name={folder}-FR-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt          # Standard output log
#SBATCH --error=error_%j.txt            # Standard error log
#SBATCH --time=24:00:00                 # Time limit (10 hours)
#SBATCH --partition=gpu-l40s            # GPU partition
#SBATCH --ntasks=1                      # One task
#SBATCH --cpus-per-task=4               # Request 4 CPU cores
#SBATCH --mem=150G                      # Memory request (150 GB)
#SBATCH --gres=gpu:4                   # Request 4 GPUs

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \\
    --data \"/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/data_configs/data.yaml\" \\
    --epochs 300 \\
    --model fasterrcnn_resnet50_fpn \\
    --name {folder}_Faster_RCNN \\
    --batch 16 \\
    --disable-wandb \\
    --imgsz 640
"

# List of folders to process
FOLDERS=("0%" "25%" "50%" "75%" "100%" "+25k" "+50k" "+90k")

# Loop over each folder and generate the Train_RCNN.sh script
for folder in "${FOLDERS[@]}"; do
    # Define the directory path for each folder
    folder_dir="${BASE_DIR}/${folder}"
    
    # Check if the folder exists
    if [[ -d "$folder_dir" ]]; then
        # Replace placeholders in the template with the actual folder name
        script_content="${SCRIPT_TEMPLATE//\{folder\}/$folder}"

        # Write the content to Train_RCNN.sh in the corresponding folder
        echo "$script_content" > "${folder_dir}/Train_RCNN.sh"
        echo "Generated Train_RCNN.sh for folder $folder at $folder_dir"
    else
        echo "Directory $folder_dir does not exist. Skipping $folder."
    fi
done
