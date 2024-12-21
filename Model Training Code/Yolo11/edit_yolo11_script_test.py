import os

# Base directory path where test_predict.py files should be created
base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/Yolo11/models/300_epochs_x"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Loop through each folder and write the test_predict.py file
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "yolo11_test.sh")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Define the template as a regular multi-line string with custom placeholders
    template = """#!/bin/bash
wandb online
yolo task=detect mode='predict' model=./runs/detect/train/weights/best.pt source='/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Test/images' imgsz=1920 save_txt=True


"""
    # Replace placeholders with actual values
    script_content = template.replace("__base_dir__", base_dir).replace("__folder__", folder)
    
    # Write the test_predict.py file
    with open(file_path, 'w') as file:
        file.write(script_content)
    
    print(f"Created {file_path}")
