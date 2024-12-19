import numpy as np
import cv2
import pandas.io.common
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas

from models.create_fasterrcnn_model import create_model
from utils.annotations import (
    inference_annotations, convert_detections
)
from utils.general import set_infer_dir
from utils.transforms import infer_transforms, resize
from utils.logging import LogJSON

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

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '-o', '--output',
        default=None, 
        help='folder path to output data',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', 
        default=0.3, 
        type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show',  
        action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', 
        dest='mpl_show', 
        action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=None,
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--square-img',
        dest='square_img',
        action='store_true',
        help='whether to use square image resize, else use aspect ratio resize'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    parser.add_argument(
        '--track',
        action='store_true'
    )
    parser.add_argument(
        '--log-json',
        dest='log_json',
        action='store_true',
        help='store a json log file in COCO format in the output directory'
    )
    parser.add_argument(
        '-t', '--table', 
        dest='table', 
        action='store_true',
        help='outputs a csv file with a table summarizing the predicted boxes'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['data'] is not None:
        with open(args['data']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    if args['output'] is not None:
        OUT_DIR = args['output']
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
    else:
        OUT_DIR=set_infer_dir() 
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE, weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            if data_configs is None:
                data_configs = True
                NUM_CLASSES = checkpoint['data']['NC']
                CLASSES = checkpoint['data']['CLASSES']
            try:
                build_model = create_model[str(args['model'])]
            except KeyError:
                build_model = create_model[checkpoint['model_name']]
            model = build_model(num_classes=NUM_CLASSES, coco_model=False)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
    else:
        raise KeyError("Checkpoint does not contain 'model_state_dict'. Please check the weights file.")

        
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] == None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = args['input']
        test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    # Define dictionary to collect boxes detected in each file 
    pred_boxes = {}
    box_id = 1

    if args['log_json']:
        log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        if args['imgsz'] != None:
            RESIZE_TO = args['imgsz']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(
            orig_image, RESIZE_TO, square=args['square_img']
        )
        height_r, width_r, channels_R = image_resized.shape
        image = image_resized.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            draw_boxes, pred_classes, scores, labels = convert_detections(
                outputs, detection_threshold, CLASSES, args
            )
            orig_image = inference_annotations(
                draw_boxes, 
                pred_classes, 
                scores,
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )

            if args['show']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()

            if args['table']:
                for box, label in zip(draw_boxes, pred_classes):
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin

                    pred_boxes[box_id] = {
                        "image": image_name,
                        "label": str(label),
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "width": width,
                        "height": height,
                        "area": width * height
                    }                    
                    box_id = box_id + 1

                df = pandas.DataFrame.from_dict(pred_boxes, orient='index')
                df = df.fillna(0)
                df.to_csv(f"{OUT_DIR}/boxes.csv", index=False)

            if args['log_json']:
                log_json.update(orig_image, image_name,scores, draw_boxes, labels, CLASSES,height_r,width_r)

        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()

    # Save JSON log file.
    if args['log_json']:
        log_json.save(os.path.join(OUT_DIR, 'log.json'))
        
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print('Path to output files: '+OUT_DIR)

if __name__ == '__main__':
    args = parse_opt()
    main(args)
