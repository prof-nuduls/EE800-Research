import os
import shutil
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Define source paths for real and synthetic data
source_real_train_images = "/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Train/images"
source_real_train_labels = "/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Train/labels"
source_real_valid_images = "/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Valid/images"
source_real_valid_labels = "/mmfs1/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Valid/labels"

source_synth_train_images = "/mmfs1/home/dmiller10/EE800 Research/Data/Synthentic Sea Drones/Compressed/Train/images"
source_synth_train_labels = "/mmfs1/home/dmiller10/EE800 Research/Data/Synthentic Sea Drones/Compressed/Train/labels"

# Define the target folder path
base_target_path = "/mmfs1/home/dmiller10/EE800 Research/Data/Yolo11_2"

# Sampling dictionary for +25k, +50k, +90k
sampling_dict = {
    "+25k": 25000,  # Add 25k synthetic images to all real images
    "+50k": 50000,  # Add 50k synthetic images to all real images
    "+90k": 90000   # Add 90k synthetic images to all real images
}

# Function to clear folder before copying
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder_path)

# Function to create the folder structure
def create_folder_structure(base_path, folder_name):
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(os.path.join(folder_path, "Train", "images"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "Train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "Valid", "images"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "Valid", "labels"), exist_ok=True)
    return folder_path

# Function to sample files
def sample_files(source_dir, num_files):
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    all_files.sort()
    return random.sample(all_files, min(num_files, len(all_files))) if num_files != "all" else all_files

# Function to copy files
def copy_files(file_list, real_source_dir, synth_source_dir, dest_folder):
    for file_name in file_list:
        source_file = os.path.join(real_source_dir if file_name in os.listdir(real_source_dir) else synth_source_dir, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        shutil.copy(source_file, dest_file)

# Main function for processing folders based on the sampling dictionary
def process_folder(task):
    folder_name, synth_count = task
    print(f"\nProcessing folder: {folder_name}")
    folder_path = create_folder_structure(base_target_path, folder_name)
    train_images_folder = os.path.join(folder_path, "Train", "images")
    train_labels_folder = os.path.join(folder_path, "Train", "labels")
    valid_images_folder = os.path.join(folder_path, "Valid", "images")
    valid_labels_folder = os.path.join(folder_path, "Valid", "labels")

    clear_folder(train_images_folder)
    clear_folder(train_labels_folder)
    clear_folder(valid_images_folder)
    clear_folder(valid_labels_folder)

    real_train_images = sample_files(source_real_train_images, "all")
    synth_train_images = sample_files(source_synth_train_images, synth_count)
    combined_train_images = real_train_images + synth_train_images
    real_train_labels = [img.replace('.png', '.txt').replace('.jpg', '.txt') for img in real_train_images]
    synth_train_labels = [img.replace('.png', '.txt').replace('.jpg', '.txt') for img in synth_train_images]
    combined_train_labels = real_train_labels + synth_train_labels

    copy_files(combined_train_images, source_real_train_images, source_synth_train_images, train_images_folder)
    copy_files(combined_train_labels, source_real_train_labels, source_synth_train_labels, train_labels_folder)

    real_valid_images = sample_files(source_real_valid_images, "all")
    real_valid_labels = [img.replace('.png', '.txt').replace('.jpg', '.txt') for img in real_valid_images]
    copy_files(real_valid_images, source_real_valid_images, "", valid_images_folder)
    copy_files(real_valid_labels, source_real_valid_labels, "", valid_labels_folder)

    print(f"Finished processing folder {folder_name}.")

# Run the main function using ProcessPoolExecutor for folder-level parallelization
if __name__ == "__main__":
    tasks = [(folder_name, synth_count) for folder_name, synth_count in sampling_dict.items()]
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(tqdm(executor.map(process_folder, tasks), total=len(tasks), desc="Processing folders"))
