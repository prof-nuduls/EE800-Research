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

## Data Download

### **Datasets**

#### **1. Synthetic Sea Drones Dataset**
- Download from [MACVi Dataset Page](https://macvi.org/dataset).
- Contains both compressed and uncompressed versions, and only Train and Valid splits.
- Folder structure:
```
Synthetic Sea Drones/
├── Compressed/
│   ├── Train/
│   │   ├── images/
│   │   ├── labels/
│   ├── Valid/
│       ├── images/
│       ├── labels/
├── Uncompressed/
    ├── Train/
    │   ├── images/
    │   ├── labels/
    ├── Valid/
        ├── images/
        ├── labels/
```

#### **2. SeaDronesSee Object Detection v2 Dataset (Real)**
- Download from [MACVi Dataset Page](https://macvi.org/dataset).
- Contains both compressed and uncompressed versions for all splits (Train, Valid, Test).
- Annotation files are stored outside the data folders in a separate `annotations/` directory.
- Folder structure:
```
SeaDronesSee Object Detection v2/
├── Compressed/
│   ├── Train/
│   │   ├── images/
│   ├── Valid/
│   │   ├── images/
│   ├── Test/
│       ├── images/
├── Uncompressed/
│   ├── Train/
│   │   ├── images/
│   ├── Valid/
│   │   ├── images/
│   ├── Test/
│       ├── images/
├── Annotations/
    ├── instances_train.json
    ├── instances_valid.json
    ├── instances_test.json
```

---

## Data Preparation

### **1. File Conversion (File Div Folder)**

#### **parse_json.py**
- Converts annotations from JSON format to VOC XML or YOLO TXT formats.
- Usage:
```bash
python parse_json.py --input-dir /path/to/json --output-dir /path/to/converted
```
- Example Output Directory after Conversion:
```
SeaDronesSee Object Detection v2/
├── Train/
│   ├── images/
│   ├── labels/
├── Valid/
│   ├── images/
│   ├── labels/
├── Test/
    ├── images/
    ├── labels/
```

### **2. Data Splitting**

#### **Train_div.py**
- Splits datasets into predefined proportions (`0%`, `25%`, `50%`, `75%`, `100%`).
- Example SLURM submission:
```bash
sbatch file_div.sh
```

#### **Train_div2.py**
- Adds synthetic data in increments (`+25k`, `+50k`, `+90k`) to the training set.
- Example SLURM submission:
```bash
sbatch run_train_div.sh
```

### **Example Output Directory for YOLO11**
```
Data/Yolo11/
├── 0%/
│   ├── Train/
│   │   ├── images/
│   │   │   ├── image1.png
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   ├── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   │       └── ...
│   ├── Valid/
│       ├── images/
│       │   ├── image1.png
│       │   ├── image2.png
│       │   └── ...
│       ├── labels/
│           ├── image1.txt
│           ├── image2.txt
│           └── ...
│   ├── data.yaml
├── 25%/
│   └── ... (same structure as 0%)
├── 50%/
│   └── ...
├── 75%/
│   └── ...
├── 100%/
│   └── ...
├── +25k/
│   └── ...
├── +50k/
│   └── ...
├── +90k/
    └── ...
```

---

## Model Training

### **YOLOv11**

#### **Configuration Example (`data.yaml`) for Faster R-CNN** for YOLOv11**
```yaml
train: ../Train/images
val: ../Valid/images
test: /base_path/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Test

nc: 5
names: ['0', '1', '2', '3', '4']
```

#### **Training Steps**

##### Individual Job Submission
- To train the YOLOv11 model for a single split, navigate to the respective folder and run:
```bash
sbatch schedule_yolo.sh
```

##### Batch Job Submission
- To batch-submit jobs for all splits, use the provided batch scripts:
  - Training:
  ```bash
  bash start_all.sh
  ```
  - Validation:
  ```bash
  bash start_all_val.sh
  ```
  - Testing:
  ```bash
  bash start_all_test.sh
  ```

#### **Results Conversion**
Convert YOLO predictions to COCO JSON format for submission:
```bash
python yolo_to_json.py --path /path/to/outputs --output results.json
```

### **Faster R-CNN**

#### **Annotations Conversion**
- Convert YOLO TXT labels to VOC XML format compatible with Faster R-CNN.
- Script: `convert_parallel.py` located in `Faster_RCNN/Yolo_to_VOC`.
- SLURM submission:
```bash
sbatch schedule.sh
```
- Output:
  - A new `annotations/` folder is added to each split in `Faster-RCNN/`.
  - Folder structure:
```
Data/Faster-RCNN/
├── 0%/
│   ├── Train/
│   │   ├── images/
│   │   ├── annotations/
│   │   ├── data_configs/
│   │       ├── data.yaml
│   ├── Valid/
│       ├── images/
│       ├── annotations/
│       ├── data_configs/
│           ├── data.yaml
├── 25%/
│   └── ...
├── 50%/
│   └── ...
├── 75%/
│   └── ...
├── 100%/
│   └── ...
├── +25k/
│   └── ...
├── +50k/
│   └── ...
├── +90k/
    └── ...
```

#### **Configuration Example (`data.yaml`) for YOLOv11 and Faster R-CNN**
```yaml
CLASSES:
- __background__
- '1'
- '2'
- '3'
- '4'
- '5'
NC: 6
SAVE_VALID_PREDICTION_IMAGES: false
TRAIN_DIR_IMAGES: /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/+50k/Train/images
TRAIN_DIR_LABELS: /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/+50k/Train/annotations
VALID_DIR_IMAGES: /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/+50k/Valid/images
VALID_DIR_LABELS: /mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/+50k/Valid/annotations
```
#### **Training Steps**

##### Individual Job Submission
- Navigate to the respective folder and run:
  ```bash
  sbatch Train_RCNN.sh
  ```
- Similarly, validation and testing can be submitted with:
  ```bash
  sbatch Val_RCNN.sh
  sbatch Test_RCNN.sh
  ```

##### Batch Job Submission
- Use the provided batch scripts to submit jobs for all splits:
  - Training:
    ```bash
    bash run_all_train.sh
    ```
  - Validation:
    ```bash
    bash run_all_val.sh
    ```
  - Testing:
    ```bash
    bash resume_all.sh
    ```

##### Example Configuration (`data.yaml`)
```yaml
train: ../Train/images
val: ../Valid/images
annotations: ../Train/annotations
num_classes: 5
save_predictions: true
```

##### Example SLURM Script (`Train_RCNN.sh`)
```bash
#!/bin/bash
#SBATCH --job-name=faster_rcnn_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python train.py --data data.yaml
```

### **RetinaNet**

#### **Annotations and Configuration**
- RetinaNet uses the same VOC XML annotations as Faster R-CNN.
- Each split contains a `config.py` file specifying training parameters.
- Example `config.py`:
```python
import torch

BATCH_SIZE = 8
RESIZE_TO = 640
NUM_EPOCHS = 300
NUM_WORKERS = 4
LR = 0.00001
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_IMG = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/Train/images'
TRAIN_ANNOT = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/Train/annotations'
VALID_IMG = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/Valid/images'
VALID_ANNOT = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/{folder}/Valid/annotations'
CLASSES = ['__background__', '1', '2', '3', '4', '5']
NUM_CLASSES = len(CLASSES)
```

#### **Training Steps**

##### Individual Job Submission
- Navigate to the respective folder and run:
```bash
sbatch Train_Retina.sh
```

##### Batch Job Submission
- Use the provided batch scripts to submit jobs for all splits:
  - Training:
    ```bash
    bash run_all_retinanet_models.sh
    ```
  - Validation:
    ```bash
    bash run_all_val_retinanet_models.sh
    ```
  - Testing:
    ```bash
    bash run_all_test_retinanet_models.sh
    ```

#### **Results Conversion**
Convert RetinaNet predictions to COCO JSON format for submission:
```bash
python yolo_to_json.py --path /path/to/outputs --output results.json
```
