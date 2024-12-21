import json
import argparse

def convert_annotations(input_path, output_path):
    # Load the input `log.json` file
    with open(input_path, 'r') as log_file:
        log_data = json.load(log_file)

    # Create a mapping of file_name to id from the "images" section
    # Placeholder for the converted annotations
    converted_annotations = []

    # Process the "annotations" section
    for annotation in log_data.get("annotations", []):
        file_name = annotation.get("image_id")
        print(file_name)
        converted_entry = {
            "image_id": int(file_name),  # Map file_name to id
            "category_id": int(annotation.get("category_id")),
            "score": float(annotation.get("score")),  # Set score to 1.0
            "bbox": annotation.get("bbox")
        }
        converted_annotations.append(converted_entry)

    # Save the converted annotations to the output file
    with open(output_path, 'w') as output_file:
        json.dump(converted_annotations, output_file, indent=4)

    print(f"Conversion complete. Saved to `{output_path}`.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert annotations from log.json format.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input log.json file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the converted annotations.")

    # Parse arguments
    args = parser.parse_args()

    # Run the conversion
    convert_annotations(args.input_path, args.output_path)

