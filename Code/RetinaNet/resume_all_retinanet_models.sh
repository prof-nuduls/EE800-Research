#!/bin/bash

# Base directory where the training folders (like 0%, 25%, etc.) are located
BASE_DIR="/mmfs1/home/dmiller10/EE800 Research/Code/RetinaNet/"

# List of folders to process
FOLDERS=("0%" "25%" "50%" "75%" "100%" "+25k" "+50k" "+90k")
FOLDERS=("75%" "100%" "+25k" "+50k" "+90k")
# Loop over each folder, cd into it, and submit the job script
for folder in "${FOLDERS[@]}"; do
    # Define the directory path for each folder
    folder_dir="${BASE_DIR}/${folder}"
    
    # Check if the folder exists and if Train_RCNN.sh is present
    if [[ -d "$folder_dir" && -f "${folder_dir}/Resume_Retina.sh" ]]; then
        # Change to the folder and submit the job
        cd "$folder_dir" || continue
        sbatch Resume_Retina.sh
        echo "Submitted job for folder $folder"
    else
        echo "Skipping $folder: Directory or Train_Retina.sh not found."
    fi
done
