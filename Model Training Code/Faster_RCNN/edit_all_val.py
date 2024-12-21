import os

# Base directory path where Train_RCNN.sh files should be created
base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/Faster_RCNN/model/models/300_epochs_full"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Template for the Train_RCNN.sh file
template = """#!/bin/bash
#SBATCH --job-name={folder}-FR-val     # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=compute                 # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores

python eval.py --model fasterrcnn_resnet50_fpn_v2 --weights /outputs/training/{folder}_Faster_RCNN_v2_p30/best_model.pth --data "/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/data_configs/data.yaml" --imgsz 1920 --batch 4 --verbose"""

# Loop through each folder and write the Train_RCNN.sh file
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "Val_RCNN.sh")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Write the Train_RCNN.sh file with the correct folder name and settings
    with open(file_path, 'w') as file:
        file.write(template.format(folder=folder))
    
    print(f"Created {file_path}")
