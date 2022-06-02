import numpy as np
import os
import cv2
import sys
from PIL import Image
import torch
import torchvision
import yaml
import json
from pathlib import Path


class Img:
    def __init__(self):
        self.name = None
        self.path = None
        self.data = None


def get_image(path=None):
    if path is None:
        raise ValueError
    img = cv2.imread(path)
    h, w, c = img.shape
    return img, (h, w, c)


def get_annotation(path=None):
    suffix = Path(path).suffix
    with open(path, 'r') as stream:
        if suffix == '.json':
            data = json.load(stream)
        if suffix == '.yaml':
            data = yaml.safe_load(stream)
        if suffix == '.txt':
            data = stream.readlines()
    return data


def get_data(img_path, lbl_path, anno_path):
    image, hwc = get_image(img_path)
    labels = get_annotation(lbl_path)

    annotations = get_annotation(anno_path)
    class_names = annotations['names']

    convert_to_absolute_coordinates = None


def main():
    get_data(img_path='./data/109.jpg', lbl_path='./data/109.txt', anno_path='./data/coco128.yaml')
    print('!')


if __name__ == "__main__":
    main()
