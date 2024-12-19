import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

def yolo_to_voc(yolo_path, voc_path, image_path, min_dimension=5):
    try:
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = ""
        filename = ET.SubElement(root, "filename")
        filename.text = os.path.basename(image_path)

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"

        has_valid_boxes = False

        if os.path.exists(yolo_path):
            with open(yolo_path, 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue  # Skip malformed lines

                class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts)

                xmin = int((center_x - bbox_width / 2.0) * image_width)
                ymin = int((center_y - bbox_height / 2.0) * image_height)
                xmax = int((center_x + bbox_width / 2.0) * image_width)
                ymax = int((center_y + bbox_height / 2.0) * image_height)
		

                # Validate bounding box dimensions
                if xmax-xmin < min_dimension or ymax-ymin < min_dimension:
                    continue

                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = str(int(class_id) + 1)
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"

                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)

                has_valid_boxes = True

        # Write the XML file even if there are no valid bounding boxes
        tree = ET.ElementTree(root)
        tree.write(voc_path)

        if not has_valid_boxes:
            print(f"No valid bounding boxes found for {yolo_path}. Empty XML created.")

    except Exception as e:
        print(f"Error processing file {yolo_path}: {e}")
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"{yolo_path}: {e}\n")

def process_file(args):
    yolo_path, voc_path, image_path = args
    if os.path.exists(image_path):
        yolo_to_voc(yolo_path, voc_path, image_path)
    else:
        print(f"Image not found for {os.path.basename(yolo_path)}, skipping.")

def process_folder(yolo_labels_path, voc_annotations_path, images_path, cpu_count=2000):
    yolo_files = os.listdir(yolo_labels_path)
    file_args = []
    for file in yolo_files:
        yolo_path = os.path.join(yolo_labels_path, file)
        voc_file = os.path.basename(file).replace('.txt', '.xml')
        voc_path = os.path.join(voc_annotations_path, voc_file)
        image_file = os.path.basename(file).replace('.txt', '.png')
        image_path = os.path.join(images_path, image_file)
        file_args.append((yolo_path, voc_path, image_path))

    # Parallel processing using specified CPUs
    with Pool(processes=cpu_count) as pool:
        list(tqdm(pool.imap_unordered(process_file, file_args), total=len(file_args), desc="Processing files"))

# Paths
yolo_base_path = "/mmfs1/home/dmiller10/EE800 Research/Data/Yolo11"
faster_rcnn_base_path = "/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']
subfolders = ['Train', 'Valid']

# Process each folder
for folder in folders:
    for subfolder in subfolders:
        yolo_labels_path = os.path.join(yolo_base_path, folder, subfolder, "labels")
        voc_annotations_path = os.path.join(faster_rcnn_base_path, folder, subfolder, "annotations")
        images_path = os.path.join(faster_rcnn_base_path, folder, subfolder, "images")

        if not os.path.exists(voc_annotations_path):
            os.makedirs(voc_annotations_path, exist_ok=True)

        if os.path.exists(yolo_labels_path):
            process_folder(yolo_labels_path, voc_annotations_path, images_path, cpu_count=2000)
        else:
            print(f"Directory not found: {yolo_labels_path}")
