import os

# Base directory path where test_predict.py files should be created
base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/Yolo11/models/300_epochs_x"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Loop through each folder and write the test_predict.py file
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "schedule_yolo_resume.sh")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Define the template as a regular multi-line string with custom placeholders
    template = """#!/bin/bash
#SBATCH --job-name=f-Yolo11x-__folder__      # Job name
#SBATCH --output=output_%j.txt          # Standard output log
#SBATCH --error=error_%j.txt            # Standard error log
#SBATCH --time=24:00:00                 # Time limit (10 hours)
#SBATCH --partition=gpu-l40s           # GPU partition
#SBATCH --ntasks=1                      # One task
#SBATCH --cpus-per-task=4               # Request 4 CPU cores
#SBATCH --mem=48G                      # Memory request (150 GB)
#SBATCH --gres=gpu:1                    # Request 4 GPUs


pip install --user ultralytics supervision roboflow
# Install the required packages for Ultralytics YOLO and Weights & Biases
srun ./yolo11_resume.sh
"""
    # Replace placeholders with actual values
    script_content = template.replace("__base_dir__", base_dir).replace("__folder__", folder)
    
    # Write the test_predict.py file
    with open(file_path, 'w') as file:
        file.write(script_content)
    
    print(f"Created {file_path}")
