import os

# Base directory path where resume_train.py files should be created
base_dir = "/mmfs1/home/dmiller10/EE800 Research/Code/RetinaNet/"
folders = ['0%', '25%', '50%', '75%', '100%', '+25k', '+50k', '+90k']

# Template for the resume_train.py file
template = """
import re
import os
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, PROJECT_NAME,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS, TRAIN_IMG, TRAIN_ANNOT,
    VALID_IMG, VALID_ANNOT, CLASSES, RESIZE_TO, BATCH_SIZE, LR, AMP, RESOLUTIONS
)
from model import create_model
from utils.general import (
    SaveBestModel, save_model, save_loss_plot, save_mAP, set_training_dir
)
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from engine import train, validate
from utils.logging import set_log, coco_log

# Initialize visualization style.
plt.style.use('ggplot')

# Function to parse `train.log` and extract metrics.
def parse_train_log(log_path):
  
    # Parse the train.log file to extract mAP@0.50:0.95 and mAP@0.50 metrics.
    #:param log_path: Path to the train.log file.
    #:return: Lists of mAP@0.50:0.95 and mAP@0.50 values.
    ap_50_95_key = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]'
    ap_50_key = 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]'

    map_50_95_list = []
    map_50_list = []

    with open(log_path, 'r') as file:
        for line in file:
            if ap_50_95_key in line:
                value = float(re.search(r"= (\d+\.\d+)", line).group(1))
                map_50_95_list.append(value)
            elif ap_50_key in line:
                value = float(re.search(r"= (\d+\.\d+)", line).group(1))
                map_50_list.append(value)
    
    return map_50_95_list, map_50_list

# Main function to run training.
if __name__ == '__main__':
    # Set up output directory and logging.
    OUT_DIR = set_training_dir(PROJECT_NAME)
    set_log(OUT_DIR)

    # Load datasets and data loaders.
    train_dataset = create_train_dataset(TRAIN_IMG, TRAIN_ANNOT, CLASSES, RESIZE_TO)
    valid_dataset = create_valid_dataset(VALID_IMG, VALID_ANNOT, CLASSES, RESIZE_TO)
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\\n")

    # Define the model and move to the computation device.
    if RESOLUTIONS:
        min_size = tuple(RESOLUTIONS[i][0] for i in range(len(RESOLUTIONS)))
        max_size = RESOLUTIONS[-1][0]
    else:
        min_size, max_size = (RESIZE_TO,), RESIZE_TO
    model = create_model(num_classes=NUM_CLASSES, min_size=min_size, max_size=max_size)
    model = model.to(DEVICE)

    # Display model parameters.
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total parameters.")
    print(f"{total_trainable_params:,} training parameters.")

    # Set up optimizer and scheduler.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)

    # Path to the last saved weights.
    last_weights_path = os.path.join(OUT_DIR, "last_model.pth")

    # Checkpoint recovery.
    start_epoch = 0
    if os.path.exists(last_weights_path):
        checkpoint = torch.load(last_weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}.")

    # Initialize metric arrays.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Parse train.log for previous metrics.
    log_file_path = os.path.join(OUT_DIR, "train.log")
    if os.path.exists(log_file_path):
        print(f"Parsing existing log file: {log_file_path}")
        map_list, map_50_list = parse_train_log(log_file_path)
        print(f"Resumed mAP@0.50:0.95: {map_list}")
        print(f"Resumed mAP@0.50: {map_50_list}")

    # Visualize transformed images if enabled.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils.general import show_tranformed_image
        show_tranformed_image(train_loader)

    # Save the best model instance.
    save_best_model = SaveBestModel()

    # Training loop.
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Start timer and run training/validation.
        train_loss = train(model, train_loader, optimizer, DEVICE, scaler=torch.cuda.amp.GradScaler() if AMP else None)
        stats = validate(model, valid_loader, DEVICE)

        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {stats[0]}")
        print(f"Epoch #{epoch+1} mAP@0.50: {stats[1]}")

        # Update metrics.
        train_loss_list.append(train_loss)
        map_list.append(stats[0])
        map_50_list.append(stats[1])

        # Save the best and current models.
        save_best_model(model, stats[0], epoch, OUT_DIR)
        save_model(epoch, model, optimizer, OUT_DIR)

        # Save loss and mAP plots.
        save_loss_plot(OUT_DIR, train_loss_list)
        save_mAP(OUT_DIR, map_50_list, map_list)

        # Log metrics to train.log.
        coco_log(OUT_DIR, stats)

        # Update scheduler.
        scheduler.step()
        print('#'*80)
"""

# Loop through each folder and write the resume_train.py file
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "resume_train.py")
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Write the resume_train.py file with the correct folder name
    with open(file_path, 'w') as file:
        file.write(template.replace("__folder__", folder))
    
    print(f"Created {file_path}")
