import json
import os
import cv2

import numpy as np

from itertools import pairwise
from torch.utils.data import Dataset


class TuSimpleColumns:
    IMG = "raw_file"
    H_SAMPLES = "h_samples"
    LANES = "lanes"
    ALL_COLUMNS = [IMG, H_SAMPLES, LANES]


class TuSimpleDataset(Dataset):
    def __init__(self, labels_file, root_dir=""):
        self._gt_img_list = []
        self._gt_lanes_list = []
        self._gt_y_list = []

        self._load_dataset(labels_file, root_dir)

    def _load_dataset(self, labels_file, root_dir):
        with open(os.path.join(root_dir, labels_file), "r") as file:
            json_gt = [json.loads(x) for x in file.readlines()]

        for element in json_gt:
            img_path = os.path.join(root_dir, element["raw_file"])
            self._gt_img_list.append(img_path)
            self._gt_lanes_list.append(element["lanes"])
            self._gt_y_list.append(element["h_samples"])

    def trim_sample(self, needed_samples):
        if needed_samples > len(self._gt_img_list):
            return
        self._gt_img_list = self._gt_img_list[:needed_samples]
        self._gt_lanes_list = self._gt_lanes_list[:needed_samples]
        self._gt_y_list = self._gt_y_list[:needed_samples]

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        img = self._get_img(idx)
        label = self._get_ground_truth(idx, img.shape[1:])
        return img, label

    def _get_img(self, idx):
        img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)
        return self._transform(img)

    def _transform(self, img):
        print(img.shape)
        img = np.transpose(img, (2, 0, 1))
        print(img.shape)
        return img / 255.0

    def _get_ground_truth(self, idx, mask_shape):
        gt_lanes = self._gt_lanes_list[idx]
        y_samples = self._gt_y_list[idx]

        gt_lanes_vis = [
            [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes
        ]
        label = np.zeros(list(mask_shape) + [1], dtype=np.float32)

        for lane in gt_lanes_vis:
            for prv_pt, pt in pairwise(lane):
                cv2.line(label, prv_pt, pt, color=255, thickness=2)

        return self._transform(label)
