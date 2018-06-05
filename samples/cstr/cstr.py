"""
Mask R-CNN
Train on the construction site images dataset and implement color splash effect.

Copyright (c) 2018 Yingge WAN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cstr.py train --dataset=/path/to/cstr/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 cstr.py train --dataset=/path/to/cstr/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 cstr.py train --dataset=/path/to/cstr/dataset --weights=imagenet

    # Run detection
    python3 cstr.py detect --dataset=/path/to/dataset --weights=<last or /path/to/weights.h5>

    # Apply color splash to an image
    python3 cstr.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 cstr.py splash --weights=last --video=<URL or path to file>
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
# import labelme
import base64
import io
import PIL.Image
import math


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/cstr/")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets", "cstr")

# Image size info
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


############################################################
#  Configurations
############################################################


class CstrConfig(Config):
    """Configuration for training on the construction site images dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cstr"

    # Train on 2 GPU and 2 images per GPU. Batch size is 4 (GPUs * images/GPU).
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 30  # Background + 30 different classes

    # Input image resizing
    # Random crops of size 512x512
    # IMAGE_RESIZE_MODE = "sqaure"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Number of training steps per epoch.
    # Use a smaller epoch if the data is simple.
    STEPS_PER_EPOCH = 1000

    # Use small validation steps since the epoch is small.
    VALIDATION_STEPS = 50
    
    
############################################################
#  Dataset
############################################################

class CstrDataset(utils.Dataset):
    
    def add_image(self, source, image_id, image, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "image": image,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        
    def load_data(self, dataset_dir, subset):
        """Load a subset of the cstr dataset.        
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """        
        
        # Class names
        class_names = [
            'worker-formwork',
            'worker-concrete',
            'worker-welding',
            'worker-rebar',
            'worker-scaffolding',
            'worker-dump',
            'worker-heavy',
            'worker-aerial',
            'worker-other',
            'worker-idle',
            'rebar-bs',
            'rebar-wc',
            'rebar-material',
            'steel',
            'concrete-pouring',
            'concrete-forming',
            'formwork-bs',
            'formwork-wc',
            'formwork-material',
            'scaffolding',
            'excavator',
            'bulldozer',
            'dump-truck',
            'concrete-bucket',
            'concrete-mixer',
            'concrete-pump',
            'tower-crane',
            'crane',
            'basket',
            'machine-other'
        ]
        # Add classes
        for i in range(len(class_names)):
            self.add_class("cstr", i + 1, class_names[i])
            
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations        
        for annotation_filename in os.listdir(dataset_dir):
            annotation = json.load(open(os.path.join(dataset_dir, annotation_filename)))
            
            # Load image
            f = io.BytesIO()
            f.write(base64.b64decode(annotation['imageData']))
            image = np.array(PIL.Image.open(f))
            
            # image=labelme.utils.img_b64_to_array(annotation['imageData'])
            height, width = image.shape[:2]
            self.add_image(
                "cstr",
                image_id=annotation_filename,
                image=image,
                width=width, height=height,
                polygons=annotation['shapes'])        
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        return self.image_info[image_id]["image"]

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one binary mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_names = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon([point[1] for point in p['points']], [point[0] for point in p['points']])
            mask[rr, cc, i] = 1
            class_names.append(p['label'])
        
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(class_name) for class_name in class_names], dtype=np.int32)

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids
    
#         shapes = image_info['shapes']
#         image_data = image_info['image']
#         count = len(shapes)
#         mask = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, count], dtype=np.bool)
#         class_names = []
#         for i, region in enumerate(shapes):
#             temp = labelme.utils.polygons_to_mask(image_data.shape, region['points'])
#             mask[:, :, i:i+1] = temp.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
#             class_names.append(region['label'])
#         return mask, class_ids


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CstrDataset()
    dataset_train.load_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CstrDataset()
    dataset_val.load_data(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=4,  # 40
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=12,  # 120
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=16,  # 160
                layers='all')
    
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
    
    
# Load and display random samples

def display_samples():
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

        
############################################################
#  Detection
############################################################

def detect(model):
    """Run detection on images in the given directory."""
    print("Running on {}".format(args.dataset))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = CstrDataset()
    dataset.load_data(args.dataset, "val")
    dataset.prepare()
    # Load over images
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            # show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))  


# There are some problems with this function
def evaluate(model, limit=0):
    """Runs construction site images dataset evaluation. Compute VOC-Style mAP @ IoU=0.5
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Limit to a subset
    if limit:
        image_ids = np.random.choice(args.dataset.image_ids, limit)
    # With no limit, use all images from args.dataset
    else:
        image_ids = args.dataset.image_ids

    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(args.dataset, inference_config,
                                image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id,
                            r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)

    print("mAP: ", np.mean(APs))

    
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on construction site images dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect' or 'splash' on construction site images")
    parser.add_argument('--dataset', required=False,
                        default=IMAGE_DIR,
                        metavar="/path/to/cstr/dataset/",
                        help='Directory of the construction site images dataset')
    parser.add_argument('--weights', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights.h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--limit', required=False,
                        default=50,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=50)')
    args = parser.parse_args()
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"    
    elif args.command == "detect":
        assert args.dataset, "Argument --dataset is required for detection"
    elif args.command == "splash":
        assert args.image, "Provide --image to apply color splash"
            
    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CstrConfig()
    else:
        class InferenceConfig(CstrConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate or splash
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect' or 'splash'".format(args.command))
