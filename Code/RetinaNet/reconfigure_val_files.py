import os

base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/RetinaNet"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Template for the Train_RCNN.sh file
template = """#!/bin/bash
#SBATCH --job-name={folder}Val-Retina-300epochs      # Job name reflecting folder/model
#SBATCH --output=output_%j.txt                # Standard output log
#SBATCH --error=error_%j.txt                  # Standard error log
#SBATCH --time=24:00:00                       # Time limit (10 hours)
#SBATCH --partition=compute                 # GPU partition
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores

python eval.py --weights /mmfs1/home/dmiller10/EE800 Research/Code/RetinaNet/{folder}/runs/training/retinanet/best_model.pth --input-images /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/Valid/images --input-annots /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/Valid/annotations
"""

# Loop through each folder and write the Train_RCNN.sh file
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "Val_Retina.sh")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Write the Train_RCNN.sh file with the correct folder name and settings
    with open(file_path, 'w') as file:
        file.write(template.format(folder=folder))
    
    print(f"Created {file_path}")
