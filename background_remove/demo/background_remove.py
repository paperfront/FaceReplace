import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

from background_remove.maskrcnn_benchmark.config import cfg
from background_remove.demo.predictor import COCODemo

def resize2SquareKeepingAspectRation(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.ones((dif, dif), dtype=img.dtype)*255
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.ones((dif, dif, c), dtype=img.dtype)*255
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

def remove_background(input_path):
    config_file = "background_remove/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    coco_demo = COCODemo(
        cfg,
        min_image_size=100,
        confidence_threshold=0.7,
    )
    predictions,isfind = coco_demo.run_on_opencv_image(image,category=1,is_crop=True)

    img = resize2SquareKeepingAspectRation(predictions, 256, cv2.INTER_AREA)
    image_bgr = img
    # get the image dimensions (height, width and channels)
    h, w, c = image_bgr.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    # create a mask where white pixels ([255, 255, 255]) are True
    white = np.all(image_bgr == [255, 255, 255], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    image_bgra[white, -1] = 0
    # save the image

    final_path = "temp_no_background.png"
    cv2.imwrite(final_path, image_bgra)
    return final_path
