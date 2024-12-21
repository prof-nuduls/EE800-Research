#!/bin/bash
wandb online
yolo task=detect mode='train' model=yolo11x.pt data='/mmfs1/home/dmiller10/EE800 Research/Data/Yolo11/100%/data.yaml' epochs=500 imgsz=640 lr0=.0001 patience=50 device=0 plots=True batch=-1
yolo task=detect mode='val' model=./runs/detect/train/weights/best.pt data='/mmfs1/home/dmiller10/EE800 Research/Data/Yolo11/100%/data.yaml'


