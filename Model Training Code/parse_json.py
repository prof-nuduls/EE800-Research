import json
import os

# File paths for JSON inputs and output directories
json_file_train = '/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/instances_train_origl.json'  # Adjust this path for the training dataset
json_file_val = '/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/instances_val_origl.json'      # Adjust this path for the validation dataset
json_file_test = '/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/instances_test_nogt_origl.json'  # Adjust this path for the test dataset
output_dir_train = '/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/test/labels/'        # Output directory for training labels
output_dir_val = '/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/val/labels/'            # Output directory for validation labels
output_dir_test = '/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/train/labels/'          # Output directory for test labels

# Create the output directories if they do not exist
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_val, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

# Function to normalize bbox coordinates (x, y, width, height) and format to 6 decimals
def normalize_bbox(bbox, image_width, image_height):
    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height
    return [round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)]

# Function to create annotation files with normalized bbox and 6 decimal precision
def create_annotation_files(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Go through each annotation
    for image in data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']

        # Find all annotations for this image_id
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        # Create a text file for the current image
        output_file = os.path.join(output_dir, f"{image_id}.txt")

        with open(output_file, 'w') as f:
            for annotation in annotations:
                class_id = annotation['category_id'] - 1  # Assuming categories start from 1
                bbox = normalize_bbox(annotation['bbox'], image_width, image_height)
                bbox_str = ' '.join(map(str, bbox))
                f.write(f"{class_id} {bbox_str}\n")

# Generate annotation files for training, validation, and test datasets
create_annotation_files(json_file_train, output_dir_train)
create_annotation_files(json_file_val, output_dir_val)
create_annotation_files(json_file_test, output_dir_test)

print(f"Annotation files have been generated in {output_dir_train}, {output_dir_val}, and {output_dir_test}.")
