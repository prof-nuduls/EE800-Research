import os
import json

# Base directory containing the YOLO label folders
base_dir = "/home/dmiller10/EE800 Research/Code/Yolo11/models/300_epochs_full"

# Loop through each subfolder
for subfolder in ["0%", "25%", "50%", "75%", "100%", "+25k", "+50k", "+90k"]:
    labels_dir = os.path.join(base_dir, subfolder, "runs", "detect", "predict3", "labels")
    output_file = os.path.join(base_dir, subfolder, f"submission_{subfolder}.json")
    
    # Initialize the list to store JSON objects
    json_data = []
    
    # Process each label file
    if os.path.exists(labels_dir):
        for label_file in os.listdir(labels_dir):
            if label_file.endswith(".txt"):
                image_id = int(label_file.split(".")[0])  # Assuming file name matches image_id
                
                # Read the YOLO label file
                with open(os.path.join(labels_dir, label_file), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        category_id = int(parts[0])
                        bbox = list(map(float, parts[2:6]))
                        score = float(parts[1]) if len(parts) > 5 else 1.0
                        
                        # Convert YOLO bbox (center_x, center_y, width, height) to required format
                        x_min = bbox[1] - bbox[3] / 2
                        y_min = bbox[2] - bbox[5] / 2
                        bbox_converted = [x_min, y_min, bbox[3], bbox[4]]
                        
                        # Append the JSON object
                        json_data.append({
                            "image_id": image_id,
                            "category_id": category_id,
                            "score": score,
                            "bbox": bbox_converted
                        })
    
    # Write JSON data to the output file
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Submission file created for {subfolder}: {output_file}")
