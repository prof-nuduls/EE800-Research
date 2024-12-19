#!/bin/bash
#!/bin/bash

yolo train resume model=./runs/detect/train/weights/last.pt 
yolo task=detect mode='val' model=./runs/detect/train/weights/best.pt data='/mmfs1/home/dmiller10/EE800 Research/Data/Yolo11/0%/data.yaml'


