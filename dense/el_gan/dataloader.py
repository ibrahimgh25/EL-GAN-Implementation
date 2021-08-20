# coding: utf-8


import os
from json import loads
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
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

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)
            gt_lanes = self._gt_lanes_list[idx]
            y_samples = self._gt_y_list[idx]

            img = img/255
            gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
            label_img = np.zeros(list(img.shape[:2]) + [1], dtype=np.float32)
            for lane in gt_lanes_vis:
                prv_pt = None
                for pt in lane:
                    if prv_pt:
                        cv2.line(label_img, prv_pt, pt, color=255, thickness=2)
                    prv_pt = pt

            # optional transformations
            if self.transform:
                img = self.transform(img)
                label_img = self.transform(label_img)

            inv_label = abs(255 - label_img)
            label_img = np.dstack((label_img, inv_label)) / 255
            # reshape for pytorch
            # tensorflow: [height, width, channels]
            # pytorch: [channels, height, width]
            img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
            label_img = np.reshape(label_img, (label_img.shape[2], label_img.shape[0], label_img.shape[1]))
            if self.transform:
                img = self.tranform(img)
                label_img = self.transform(img)
            return img, label_img
        except:
            return tensor(1), tensor(1)
        
if __name__ == '__main__':
    from shutil import copy
    def copy_img(path_):
        # old_path = os.path.join(root_dir, path_)
        new_path = path_.replace('Personal/Final Year Project/TuSimple Dataset', 'Mini Dataset')
        print(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path.replace('/20.jpg', ''))
        copy(path_, new_path)
        new_path = new_path.replace(r'C:\Users\user\Desktop\Personal\Final Year Project\\', '')
        return new_path
    
    json_path = r'label_data_mini.json'
    root_dir = r'C:/Users/user/Desktop/Personal/Final Year Project/TuSimple Dataset/Train Set'
    dataset = LaneDataSet(json_path, root_dir)

    img, label = dataset.__getitem__(1)
