# EE800-Research
**Derick Miller**  
**Dept. of Electrical and Computer Engineering**  
**Stevens Institute of Technology**  
**EE800 Final Project**  

---

## Synthetic Data and Object Detection for Maritime Search and Rescue

This repository provides all the necessary resources for processing datasets, training object detection models, and evaluating their performance for maritime search and rescue (SAR) applications. By integrating synthetic data, this project aims to address the scarcity of high-quality real-world training data, enabling improved accuracy and robustness of object detection models.

The project is designed for researchers utilizing the JARVIS HPC cluster but can be adapted for any SLURM-managed server or HPC environment.

---
## Project Structure

### **1. Data Preparation**  
This folder contains scripts for preparing and converting datasets:
- **`parse_json.py`**: Converts annotations in JSON format to VOC XML or YOLO TXT formats as required by different models.
- **Steps**:
  1. Download the real dataset (SeaDroneSee) and synthetic dataset (SyntheticSeaDroneSee) from [MACVi](https://macvi.org/dataset).
  2. Place the downloaded files into the `Data` directory.
  3. Run `parse_json.py` to convert JSON labels to the required format. Example:
     ```bash
     python parse_json.py --input-dir /path/to/json --output-dir /path/to/converted
     ```

### **2. Data Splitting and Sampling**  
This folder contains scripts for splitting and sampling images and labels into respective data splits:
- **`Train_div.py`**: Handles the creation of `0%`, `25%`, `50%`, `75%`, and `100%` data splits by sampling real and synthetic data.
  - Example SLURM job submission:
    ```bash
    sbatch file_div.sh
    ```
- **`Train_div2.py`**: Handles the creation of `+25k`, `+50k`, and `+90k` data splits, adding specific amounts of synthetic data to all available real data.
  - Example SLURM job submission:
    ```bash
    sbatch run_train_div.sh
    ```
- **`count.sh`**: Counts files with and without underscores in filenames within a specific directory for validation purposes.
  - Example:
    ```bash
    bash count.sh
    ```

### **3. Model Training Code**  
Contains scripts for training models, including YOLO, Faster R-CNN, and RetinaNet. Each folder includes:
- Custom configuration files specific to each architecture (e.g., `yolov11.yaml`, `faster_rcnn.yaml`, `retinanet.yaml`).
- SLURM job submission scripts.
- Preprocessing and evaluation utilities.

---

## Example Folder Structure

Below is an example of the folder structure for data splits and the organization of images and labels:

```
Data/
├── Yolo11/
│   ├── 0%/
│   │   ├── Train/
│   │   │   ├── images/
│   │   │   │   ├── image1.png
│   │   │   │   ├── image2.png
│   │   │   │   └── ...
│   │   │   ├── labels/
│   │   │   │   ├── image1.txt
│   │   │   │   ├── image2.txt
│   │   │   │   └── ...
│   │   ├── Valid/
│   │       ├── images/
│   │       │   ├── image1.png
│   │       │   ├── image2.png
│   │       │   └── ...
│   │       ├── labels/
│   │           ├── image1.txt
│   │           ├── image2.txt
│   │           └── ...
│   ├── 25%/
│   │   └── ... (same structure as 0%)
│   ├── 50%/
│   │   └── ...
│   ├── 75%/
│   │   └── ...
│   ├── 100%/
│       └── ...
│   ├── +25k/
│       └── ...
│   ├── +50k/
│       └── ...
│   ├── +90k/
│       └── ...
```

---

## Getting Started

### **Requirements**  
- **Python 3.8+**: For all scripts.  
- **SLURM-Managed Server**: Recommended for large-scale model training.  
- **Dependencies**: Install using `pip install -r requirements.txt`.  

### **Steps**  

1. **Download and Prepare Datasets**:  
   - Download SeaDroneSee from [MACVi](https://macvi.org/dataset).  
   - Download SyntheticSeaDroneSee.  
   - Convert JSON annotations using `parse_json.py`.

2. **Split and Sample Data**:  
   - Use `Train_div.py` for standard splits (`0%`, `25%`, etc.).
   - Use `Train_div2.py` for extended splits (`+25k`, `+50k`, etc.).
   - Submit jobs using `file_div.sh` and `run_train_div.sh` as required.  

3. **Train Models**:  
   - Navigate to `Model Training Code`.  
   - Update paths in the respective `.yaml` files for each architecture (e.g., `yolov11.yaml`, `faster_rcnn.yaml`, `retinanet.yaml`).
   - Run training scripts using the provided SLURM submission files.  

4. **Evaluate Models**:  
   - Use the test scripts in each model folder to get predictions.  
   - Submit test set predictions to [MACVi](https://macvi.org) for benchmarking.  

---

## Supported Architectures

### **1. YOLO**  
Implemented using the [Ultralytics YOLOv11 pipeline](https://docs.ultralytics.com/).  


**Example Configuration (`yolov11.yaml`)**:
```yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 5  # number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

### **2. Faster R-CNN**  
Based on the [Faster R-CNN PyTorch Training Pipeline](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline).  
**Changes**:
- Changed weight loading mechanism 
- Changed prediction file outputs
**Example Configuration (`faster_rcnn.yaml`)**:
```yaml
train: /path/to/train/images
val: /path/to/val/images
num_classes: 6  # 5 object classes + background
```

### **3. RetinaNet**  
Based on the [RetinaNet Detection Pipeline](https://github.com/sovit-123/retinanet_detection_pipeline).  
**Changes**:
- Changed weight loading mechanism 
- Changed prediction file outputs
**Example Configuration (`retinanet.yaml`)**:
```yaml
train: /path/to/train/images
val: /path/to/val/images
num_classes: 5
anchor_sizes: [32, 64, 128, 256, 512]
```

---

## Example SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=train_yolo
#SBATCH --output=logs/yolo_train.out
#SBATCH --error=logs/yolo_train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python train.py --data yolov11.yaml --epochs 300 --img-size 640
```

---

## References

1. Faster R-CNN Training Pipeline: [GitHub](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline)  
2. YOLO Documentation: [Ultralytics](https://docs.ultralytics.com/)  
3. RetinaNet Detection Pipeline: [GitHub](https://github.com/sovit-123/retinanet_detection_pipeline)  

---
