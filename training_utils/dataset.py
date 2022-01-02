import os
from json import loads
from torch.utils.data import Dataset
import cv2
import numpy as np
from torch import tensor


class LaneDataSet(Dataset):
    def __init__(self, dataset, root_dir='', transform=None):
        self._gt_img_list = []
        self._gt_lanes_list = []
        self._gt_y_list = []
        self.transform = transform

        dataset = os.path.join(root_dir, dataset)
        
        self._dataset_kind = None
        self.multiprocessing_context = None

        json_gt = [loads(line) for line in open(dataset)]
        for element in json_gt:
            img_path = os.path.join(root_dir, element['raw_file'])
            self._gt_img_list.append(img_path)
            self._gt_lanes_list.append(element['lanes'])
            self._gt_y_list.append(element['h_samples'])
        assert len(self._gt_img_list) == len(self._gt_y_list) == len(self._gt_lanes_list)

    def trim_sample(self, needed_samples):
        ''' Trim the size of the dataset to have a specific number of samples'''
        if needed_samples > len(self._gt_img_list):
            return # If the needed samples are moret than the available, do nothing
        self._gt_img_list = self._gt_img_list[:needed_samples]
        self._gt_lanes_list = self._gt_lanes_list[:needed_samples]
        self._gt_y_list = self._gt_y_list[:needed_samples]
    
    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)
            gt_lanes = self._gt_lanes_list[idx]
            y_samples = self._gt_y_list[idx]


            gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
            label_img = np.zeros(list(img.shape[:2]) + [1], dtype=np.float32)
            # We use the series of points to draw the lines that define the lane markings
            for lane in gt_lanes_vis:
                prv_pt = None
                for pt in lane:
                    if prv_pt:
                        cv2.line(label_img, prv_pt, pt, color=1, thickness=2)
                    prv_pt = pt
            # Use 
            if self.transform is not None:
                augmentations = self.transform(image=img, mask=label_img)
                img = augmentations["image"]
                label_img = augmentations["mask"]
            return img, label_img
        except:
            # If an error occurs with retrieving the data, return one tensors
            # This error should be handled for in the trainging phase
            return tensor(1), tensor(1)