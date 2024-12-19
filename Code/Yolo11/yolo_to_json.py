from pathlib import Path
import imagesize
import argparse
import json


# Define categories based on your dataset
classes = ["1", "2", "3", "4", "5"]

def parse_yolo_results(path):
    path = Path(path)
    predictions = []
    
    # Collect all image files
    image_files = sorted(path.rglob("*.jpg")) + sorted(path.rglob("*.jpeg")) + sorted(path.rglob("*.png"))

    for image_file in image_files:
        # Get image dimensions
        w, h = imagesize.get(image_file)
        image_id = int(image_file.stem)  # Use image name (stem) as ID
        
        # Look for the corresponding YOLO label file
        label_file = image_file.parent / "labels" / f"{image_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            # Parse each line in the YOLO label file
            for line in lines:
                data = line.strip().split()
                category_id = int(data[0]) + 1  # YOLO starts with 0, COCO starts with 1
                x_center = float(data[1]) * w
                y_center = float(data[2]) * h
                width = float(data[3]) * w
                height = float(data[4]) * h
                score = float(data[5]) if len(data) > 5 else 1.0  # Include score if provided
                
                # Convert YOLO bbox (center, width, height) to COCO bbox (x_min, y_min, width, height)
                x_min = x_center - (width / 2)
                y_min = y_center - (height / 2)
                
                # Add the prediction to the list
                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "score": score,
                    "bbox": [x_min, y_min, width, height],
                })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO predictions to COCO JSON format")
    parser.add_argument("--path", required=True, type=str, help="Path to the directory containing images and YOLO labels")
    parser.add_argument("--output", required=True, type=str, help="Path to save the output JSON file")
    args = parser.parse_args()

    # Parse the YOLO results
    predictions = parse_yolo_results(args.path)
    
    # Save the predictions to a JSON file
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"COCO JSON saved to {args.output}")


if __name__ == "__main__":
    main()
