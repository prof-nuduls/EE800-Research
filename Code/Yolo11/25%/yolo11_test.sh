#!/bin/bash
wandb online
yolo task=detect mode='predict' model=./runs/detect/train/weights/best.pt source='/home/dmiller10/EE800 Research/Data/SeaDronesSee Object Detection v2/Uncompressed Version/Test/images' save_txt=True


