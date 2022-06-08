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
import math
from PIL import Image

WINDOW_SIZE = (200, 200)  # (Width, Height)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


class Img:
    def __init__(self):
        self.name = None
        self.path = None
        self.data = None


def get_image(path=None):
    if path is None:
        raise ValueError
    img = cv2.imread(path)
    # img = np.transpose(img, (2, 0, 1))
    return img


def read_file(path=None):
    suffix = Path(path).suffix
    with open(path, 'r') as stream:
        if suffix == '.json':
            data = json.load(stream)
        if suffix == '.yaml':
            data = yaml.safe_load(stream)
        if suffix == '.txt':
            data = stream.readlines()
    return data


def get_labels(lbl_path, style='yolo', img_size=None):
    data = read_file(path=lbl_path)

    if style == 'yolo':
        data = [[float(element) for element in line.strip().split()] for line in data]
        data = np.asarray(data)
        return data

    if style == 'absolute':
        labels = data[:, 0]
        boxes = data[:, 1:]
        return boxes, labels


def get_data(img_path, lbl_path, anno_path):
    image = get_image(img_path)
    annotations = get_labels(lbl_path)

    c, h, w = image.shape
    boxes = np.copy(annotations[:, 1:])
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h

    labels = annotations[:, 0]
    general_data = read_file(anno_path)
    class_names = general_data['names']


def draw_bboxes(image, bboxes, labels):
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy().astype(np.int16)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().astype(np.int16)

    for bbox, label in zip(bboxes[:, -4:], labels):
        cv2.rectangle(image, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        cv2.putText(image, f'{label}', (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    return image


def sliding_window_approach(image, boxes, window_size, style='yolo', iou_threshold=0.6, adaptive_mode=False,
                            debug=True):
    data_pack = dict(image_pack=None, labels_pack=None, positional_encoder=None)
    positional_encoder = {}

    img_h, img_w = image.shape[:2]
    window_width, window_height = window_size

    steps_h = math.ceil(img_h / window_height)
    steps_w = math.ceil(img_w / window_width)

    width_window_left = int(window_width / 2)
    width_window_right = int(img_w - window_width / 2)
    stride_w = int((width_window_right - width_window_left) / steps_w)

    height_window_top = int(window_height / 2)
    height_window_bot = int(img_h - window_height / 2)
    stride_h = int((height_window_bot - height_window_top) / steps_h)

    window_configs = dict(
        steps_w=steps_w,
        steps_h=steps_h,
        stride_w=stride_w,
        stride_h=stride_h,
        width_window_left=width_window_left,
        width_window_right=width_window_right,
        height_window_top=height_window_top,
        height_window_bot=height_window_bot,
    )

    image_pack = []
    if boxes is not None:
        labels_pack = []
        if style == 'yolo':
            labels = boxes[:, 0].astype(int)
            bboxes = np.round(xywh2xyxy(boxes[:, 1:]) * [img_w, img_h, img_w, img_h]).astype(np.uint16)
            image = draw_bboxes(image, bboxes, labels)

    for i in range(steps_h + 1):
        for j in range(steps_w + 1):
            wcw = width_window_left + j * stride_w  # wcw - window center width
            wch = height_window_top + i * stride_h  # wch - window center height
            top = int(wch - window_height / 2)
            bot = int(wch + window_height / 2)
            left = int(wcw - window_width / 2)
            right = int(wcw + window_width / 2)

            positional_encoder[f'{i}{j}'] = {
                'top': top, 'bot': bot,
                'left': left, 'right': right,
                'window_center_x': wcw, 'window_center_y': wch
            }

            if boxes is not None:  # ANNOTATION
                cropped_labels = []
                for bbox in bboxes:
                    intersection_x_min = max(left, bbox[0])
                    intersection_y_min = max(top, bbox[1])
                    intersection_x_max = min(right, bbox[2])
                    intersection_y_max = min(bot, bbox[3])

                    intersection_x = (intersection_x_max - intersection_x_min)
                    intersection_y = (intersection_y_max - intersection_y_min)

                    if intersection_x > 0 and intersection_y > 0:
                        corrected_coords = [intersection_x_min - left, intersection_y_min - top,
                                            intersection_x_max - left, intersection_y_max - top]

                        cropped_labels.append(corrected_coords)
                # IMAGE
                if len(cropped_labels) > 0:

                    image_pack.append(image[top:bot, left:right, :])
                    labels_pack.append(np.stack(cropped_labels))

                    if debug:
                        with open(f'./data/{i}{j}.json', 'w') as stream:
                            json.dump(np.stack(cropped_labels).tolist(), stream)
                        cv2.imwrite(filename=f'./data/{i}{j}.png', img=image[top:bot, left:right, :])
            else:
                # IMAGE
                image_pack.append(image[top:bot, left:right, :])

    data_pack['image_pack'] = np.stack(image_pack)
    data_pack['positional_encoder'] = positional_encoder
    if boxes is not None:
        data_pack['labels_pack'] = labels_pack

    return data_pack, window_configs


class SlidingWindow:
    def __init__(self, window_size, style='yolo', iou_threshold=0.6, adaptive_mode=False, debug=False):
        self.style = style
        self.iou_threshold = iou_threshold
        self.adaptive_mode = adaptive_mode
        self.debug = debug
        self.positional_encoder = None

        # WINDOW attributes
        self.img_width = None
        self.img_height = None

        self.window_width, self.window_height = window_size

        self.steps_x = None
        self.steps_y = None

        self.window_left_bound = None
        self.window_right_bound = None
        self.stride_x = None

        self.window_top_bound = None
        self.window_bot_bound = None
        self.stride_y = None

    def get_img_shapes(self, image):
        if isinstance(image, torch.Tensor):
            self.img_width, self.img_height = image.shape[-2:]
        if isinstance(image, np.ndarray):
            self.img_height, self.img_width = image.shape[:2]

    def count_steps(self, image):
        if self.img_width is None:
            self.get_img_shapes(image)

        self.steps_x = math.ceil(self.img_width / self.window_width)
        self.steps_y = math.ceil(self.img_height / self.window_height)

    def count_horizontal_bounds(self):
        self.window_left_bound = int(self.window_width / 2)
        self.window_right_bound = int(self.img_width - self.window_width / 2)
        self.stride_x = int((self.window_right_bound - self.window_left_bound) / self.steps_x)

    def count_vertical_bounds(self):
        self.window_top_bound = int(self.window_height / 2)
        self.window_bot_bound = int(self.img_height - self.window_height / 2)
        self.stride_y = int((self.window_bot_bound - self.window_top_bound) / self.steps_y)

    def get_window_configs(self):
        return dict(
            steps_x=self.steps_x,
            steps_y=self.steps_y,
            stride_x=self.stride_x,
            stride_y=self.stride_y,
            window_left_bound=self.window_left_bound,
            window_right_bound=self.window_right_bound,
            window_top_bound=self.window_top_bound,
            window_bot_bound=self.window_bot_bound,
        )

    def refresh_attributes(self):
        self.positional_encoder = {}

    def init_attributes(self, image):
        self.get_img_shapes(image)  # width, height
        self.count_steps(image)  # steps = x, y
        self.count_horizontal_bounds()
        self.count_vertical_bounds()

    def one_iter_cycle(self):
        pass

    def __call__(self, image, boxes=None):
        self.positional_encoder = {}
        self.init_attributes(image)

        data_pack = dict(image_pack=None, labels_pack=None, positional_encoder=None)
        image_pack = []
        if boxes is not None:
            labels_pack = []
            if self.style == 'yolo':
                if isinstance(boxes, np.ndarray):
                    labels = boxes[:, 0]
                    bboxes = bboxes = np.round(
                        xywh2xyxy(boxes[:, 1:]) * [self.img_width, self.img_height,
                                                   self.img_width, self.img_height]).astype(np.uint16)
                    if self.debug:
                        image = draw_bboxes(image, bboxes, labels)

                if isinstance(boxes, torch.Tensor):
                    image_number = boxes[:, 0]
                    labels = boxes[:, 1]
                    bboxes = boxes[:, 2:]

                    bboxes = xywh2xyxy(boxes[:, 2:])
                    bboxes[:, [0, 2]] *= self.img_width
                    bboxes[:, [1, 3]] *= self.img_height
                    bboxes = bboxes.type(torch.int16)

                    if self.debug:
                        image = draw_bboxes(image, bboxes, labels)
                    bboxes = torch.cat([image_number[:, None], labels[:, None], bboxes], dim=1)

            else:
                raise 'Work in Progress'
                pass

        for i in range(self.steps_y + 1):
            for j in range(self.steps_x + 1):
                wcw = self.window_left_bound + j * self.stride_x  # wcw - window center width
                wch = self.window_top_bound + i * self.stride_y  # wch - window center height
                top = int(wch - self.window_height / 2)
                bot = int(wch + self.window_height / 2)
                left = int(wcw - self.window_width / 2)
                right = int(wcw + self.window_width / 2)

                self.positional_encoder[f'{i}{j}'] = {
                    'top': top, 'bot': bot,
                    'left': left, 'right': right,
                    'window_center_x': wcw, 'window_center_y': wch
                }

                if boxes is not None:  # ANNOTATION
                    cropped_labels = []
                    for bbox in bboxes:
                        intersection_x_min = max(left, bbox[-4])
                        intersection_y_min = max(top, bbox[-3])
                        intersection_x_max = min(right, bbox[-2])
                        intersection_y_max = min(bot, bbox[-1])

                        intersection_x = (intersection_x_max - intersection_x_min)
                        intersection_y = (intersection_y_max - intersection_y_min)

                        if intersection_x > 0 and intersection_y > 0:
                            corrected_coords = [bbox[0], bbox[1],
                                                intersection_x_min - left, intersection_y_min - top,
                                                intersection_x_max - left, intersection_y_max - top]

                            cropped_labels.append(corrected_coords)
                    # IMAGE
                    if len(cropped_labels) > 0:
                        if isinstance(image, np.ndarray):
                            img = image[top:bot, left:right, :]
                        if isinstance(image, torch.Tensor):
                            img = image[:, :, top:bot, left:right]

                        image_pack.append(img)

                        labels_pack.append(np.stack(cropped_labels))

                        if self.debug: #  and isinstance(image, np.ndarray):
                            with open(f'./data/{i}{j}.json', 'w') as stream:
                                json.dump(np.stack(cropped_labels).tolist(), stream, indent=4)
                            cv2.imwrite(filename=f'./data/{i}{j}.png', img=image[top:bot, left:right, :])
                else:
                    # IMAGE
                    image_pack.append(image[top:bot, left:right, :])

        if isinstance(image, torch.Tensor):
            image_pack = torch.cat(image_pack, dim=0)

        elif isinstance(image, np.ndarray):
            image_pack = np.stack(image_pack)

        data_pack['image_pack'] = image_pack
        data_pack['positional_encoder'] = self.positional_encoder
        if boxes is not None:
            data_pack['labels_pack'] = labels_pack
        return data_pack, self.get_window_configs()


def main():
    img_path = './data/109.jpg'
    # lbl_path = './data/109.txt'
    lbl_path = './data/tensor_109.txt'
    anno_path = './data/coco128.yaml'

    image = get_image(img_path)
    boxes = get_labels(lbl_path)
    boxes = boxes if lbl_path == './data/109.txt' else torch.from_numpy(boxes)
    # rez = sliding_window_approach(image=image, boxes=boxes, window_size=WINDOW_SIZE)
    sw = SlidingWindow(window_size=WINDOW_SIZE, debug=True)
    data_pack, window_configs = sw(image, boxes)


if __name__ == "__main__":
    main()
    # img = torch.Size([2, 3, 640, 640])
    # target = torch.Size([30, 6])
