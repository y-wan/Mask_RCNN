#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import labelme
from config import Config
import utils
import math
import model as modellib
import visualize


# Root directory of the project (parent directory of current directory)
ROOT_DIR = os.path.dirname(os.getcwd())

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets", "hust_dataset")

# Image size info
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Class names
CLASS_NAMES = [
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


# Configurations

class CstrConfig(Config):
    """Configuration for training on the construction site images dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cstr"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 30  # background + 30 different labels

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

#     # Use smaller anchors because our image and objects are small
#     RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

#     # Reduce training ROIs per image because the images are small and have
#     # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
#     TRAIN_ROIS_PER_IMAGE = 32

#     # Use a small epoch since the data is simple
#     STEPS_PER_EPOCH = 1000

#     # use small validation steps since the epoch is small
#     VALIDATION_STEPS = 50


# Dataset

class CstrDataset(utils.Dataset):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """
    def add_image(self, source, image_id, image, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "image": image,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        
    def load_data(self, image_dir, class_names, train=True, val_rate=0.3):
        """Load a subset of the COCO dataset.
        image_dir: The root directory of the construction site images dataset.
        val_rate: The portion of validation set of the whole dataset
        """
        num_image = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        
        if train:
            id_start = 1
            id_end = int(num_image * (1 - val_rate)) + 1
        else:
            id_start = int(num_image * (1 - val_rate)) + 1
            id_end = num_image + 1
        
        for image_id in range(id_start, id_end):
            data = json.load(open(os.path.join(IMAGE_DIR, '{0:05d}'.format(image_id) + '.json')))
            self.add_image(
                "cstr",
                image_id=image_id,
                image=labelme.utils.img_b64_to_array(data['imageData']),
                shapes=data['shapes'])

        # Add classes
        for i in range(len(class_names)):
            self.add_class("cstr", i, class_names[i])
        
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        return self.image_info[image_id]["image"]

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        shapes = image_info['shapes']
        image_data = image_info['image']
        count = len(shapes)
        mask = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, count], dtype=np.bool)
        class_names = []
        for i, region in enumerate(shapes):
            temp = labelme.utils.polygons_to_mask(image_data.shape, region['points'])
            mask[:, :, i:i+1] = temp.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
            class_names.append(region['label'])
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(class_name) for class_name in class_names])
        return mask, class_ids


# Load and display random samples

def display_samples():
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Detection

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def detect():
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                            image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # visualize labels
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(8, 8))

    # visulize detection results
    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax=get_ax())


def evaluate_cstr(model, dataset, limit=0):
    """Runs construction site images dataset evaluation. Compute VOC-Style mAP @ IoU=0.5
    dataset: A Dataset object with valiadtion data
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Limit to a subset
    if limit:
        image_ids = np.random.choice(dataset.image_ids, limit)
    # With no limit, use all images from dataset
    else:
        image_ids = dataset.image_ids

    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
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


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on construction site images dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on construction site images")
    parser.add_argument('--dataset', required=False,
                        default=IMAGE_DIR,
                        metavar="/path/to/annotations/",
                        help='Directory of the construction site images dataset')
    parser.add_argument('--model', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights.h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=MODEL_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=50,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=50)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
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
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load and load weights
    print("Loading weights ", args.model)
    if args.model.lower() == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    else if args.model.lower() == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else if args.model.lower() == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    else:
        model.load_weights(args.model, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset
        dataset_train = CstrDataset()
        dataset_train.load_data(IMAGE_DIR, CLASS_NAMES, train=True)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CstrDataset()
        dataset_val.load_data(IMAGE_DIR, CLASS_NAMES, train=False)
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
        
        # Save weights
        # Typically not needed because callbacks save after every epoch
        # Uncomment to save manually
        # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
        # model.keras_model.save_weights(model_path)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CstrDataset()
        dataset_val.load_data(IMAGE_DIR, CLASS_NAMES, train=False)
        dataset_val.prepare()
        print("Running Construction image evaluation on {} images.".format(args.limit))
        evaluate_cstr(model, dataset_val, limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
