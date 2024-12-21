#!/bin/bash

# Define the base source and destination paths
source_base="/mmfs1/home/dmiller10/EE800 Research/Code/Faster_RCNN/model"
destination_base="/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN"

# Loop through each folder in the source base directory
for folder in "$source_base"/*; do
    # Extract the folder name
    folder_name=$(basename "$folder")
    
    # Define the source and destination paths for data.yaml
    source_path="$source_base/$folder_name/data.yaml"
    destination_path="$destination_base/$folder_name/data_configs/data.yaml"
    
    # Ensure the destination directory exists
    mkdir -p "$(dirname "$destination_path")"
    
    # Move the data.yaml file
    if [ -f "$source_path" ]; then
        mv "$source_path" "$destination_path"
        echo "Moved $source_path to $destination_path"
    else
        echo "No data.yaml found in $source_path"
    fi
done
