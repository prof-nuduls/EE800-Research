import torch

BATCH_SIZE = 8 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Base image resolution transforms.
NUM_EPOCHS = 300 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.
LR = 0.00001 # Initial learning rate. 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Keep `resolutions=None` for not using multi-resolution training,
# else it will be 50% lower than base `RESIZE_TO`, then base `RESIZE_TO`, 
# and 50% higher than base `RESIZE_TO`
RESOLUTIONS = [
    (int(RESIZE_TO/2), int(RESIZE_TO/2)), 
    (int(RESIZE_TO/1.777), int(RESIZE_TO/1.777)), 
    (int(RESIZE_TO/1.5), int(RESIZE_TO/1.5)), 
    (int(RESIZE_TO/1.333), int(RESIZE_TO/1.333)), 
    (RESIZE_TO, RESIZE_TO), 
    (int(RESIZE_TO*1.333), int(RESIZE_TO*1.333)), 
    (int(RESIZE_TO*1.5), int(RESIZE_TO*1.5)), 
    (int(RESIZE_TO*1.777), int(RESIZE_TO*1.777)), 
    (int(RESIZE_TO*2), int(RESIZE_TO*2))
]
# RESOLUTIONS = None

# Training images and XML files directory.
TRAIN_IMG = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/75%/Train/images'
TRAIN_ANNOT = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/75%/Train/annotations'
# Validation images and XML files directory.
VALID_IMG = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/75%/Valid/images'
VALID_ANNOT = '/mmfs1/home/dmiller10/EE800 Research/Data/Faster-RCNN/75%/Valid/annotations'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__',
    '1',
    '2',
    '3',
    '4',
    '5',   
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Automatic Mixed Preicision?
AMP = True

# If kept None, it will be incremental as exp1, exp2,
# else it will be name provided.
PROJECT_NAME = 'retinanet' 