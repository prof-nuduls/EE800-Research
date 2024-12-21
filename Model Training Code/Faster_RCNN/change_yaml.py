import os
import yaml

# Define the base directory where the Faster-RCNN data folders are located
base_dir = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN'

# Define the new classes list
new_classes = [
    '__background__',
    '1',
    '2',
    '3',
    '4',
    '5'
]

# Iterate through each subdirectory in the base directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == 'data.yaml':
            file_path = os.path.join(root, file)
            
            # Load the existing data.yaml file
            with open(file_path, 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)
            
            # Update the classes in the data configuration
            data['CLASSES'] = new_classes
            data['NC'] = len(new_classes)  # Update the number of classes
            
            # Save the modified data.yaml file
            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file)
            
            print(f"Updated classes in {file_path}")
