import os

# Base directory path where Train_RCNN.sh files are located
base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/Faster_RCNN/model/models/300_epochs"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Loop through each folder
for folder in folders:
    file_path = os.path.join(base_dir, folder, "Train_RCNN.sh")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Modify the lines as needed
        modified_lines = []
        for line in lines:
            # Remove any line with --mem
            if "--mem=" in line:
                continue  # Skip this line entirely

            # Add mem-per-gpu if it's not already present
            if "#SBATCH --gres=gpu:4" in line:
                modified_lines.append("#SBATCH --mem-per-gpu=48G\n")
            
            # Update the model to fasterrcnn_resnet50_fpn_v2
            if "--model fasterrcnn_resnet50_fpn" in line:
                line = line.replace("fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2")
            
            # Add _v2 to the name
            if "--name" in line:
                line = line.replace("--name ", "--name " + folder + "_Faster_RCNN_v2 ")
            
            modified_lines.append(line)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)
        
        print(f"Updated {file_path}")
    else:
        print(f"{file_path} does not exist.")
