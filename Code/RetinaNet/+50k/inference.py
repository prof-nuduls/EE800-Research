import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

from model import create_model
from torchvision import transforms as transforms
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
from utils.annotations import inference_annotations

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--weights',
    default='outputs/best_model.pth',
    help='path to the model weights'
)
parser.add_argument(
    '-i', '--input', 
    help='path to input image directory or a single image',
    required=True
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
parser.add_argument(
    '--show', 
    action='store_true',
    help='whether to visualize the results in real-time'
)
parser.add_argument(
    '-nlb', '--no-labels',
    dest='no_labels',
    action='store_true',
    help='do not show labels during on top of bounding boxes'
)
args = parser.parse_args()

# Output directories.
OUT_DIR = 'outputs/inference_outputs/images'
LABELS_DIR = 'outputs/inference_outputs/labels'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# RGB format.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(args.weights, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

DIR_TEST = args.input
test_images = collect_all_images(DIR_TEST)
print(f"Test instances: {len(test_images)}")
for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    # Save original image dimensions for normalization
    original_height, original_width = orig_image.shape[:2]

    # Resize if specified
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz, args.imgsz))

    print(f"Processed image shape: {image.shape}")
    
    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply transforms
    image_input = infer_transforms(image)
    # Add batch dimension.
    image_input = torch.unsqueeze(image_input, 0)

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current FPS
    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    # Load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    # Save predictions in YOLO format
    labels_file = os.path.join(LABELS_DIR, f"{image_name}.txt")
    with open(labels_file, 'w') as f:
        if len(outputs[0]['boxes']) != 0:
            for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                # Extract and normalize bounding box coordinates
                x_min, y_min, x_max, y_max = box.tolist()
                x_center = ((x_min + x_max) / 2.0) / 640# Normalize by width
                y_center = ((y_min + y_max) / 2.0) / 640  # Normalize by height
                width = (x_max - x_min) / 640  # Normalize width
                height = (y_max - y_min) / 640  # Normalize height

                class_id = label.item()
                confidence = score.item()

                # Write YOLO format: class_id x_center y_center width height confidence
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.4f}\n")

    # Draw the bounding boxes and write the class name on top of it.
    if len(outputs[0]['boxes']) != 0:
        orig_image = inference_annotations(
            outputs, 
            args.threshold, 
            CLASSES, 
            COLORS, 
            orig_image, 
            image,
            args
        )

    if args.show:
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
    cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done... Saved labels to {labels_file}")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
