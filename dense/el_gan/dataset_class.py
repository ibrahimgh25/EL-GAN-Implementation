import os, io
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from json import loads

class TrueSimpleSet(Dataset):
    def __init__(self, labels_json, root_dir, transform=None):
        json_gt = [loads(line) for line in open(labels_json)]
        self.lane_markings = pd.DataFrame(json_gt)
        print(self.lane_markings.columns)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lane_markings)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.lane_markings.iloc[idx, 2])
        image = io.imread(img_path)
        lane_markings, y_samples = self.lane_markings.iloc[idx, [0, 1]]

        complete_output = []
        for gt_lanes, y_sample in zip(lane_markings, y_samples):
            output_img = [[(x, y) for (x, y) in zip(lane, y_sample) if x >= 0] for lane in gt_lanes]
            complete_output.append([[(x, y) for (x, y) in zip(lane, y_sample) if x >= 0] for lane in gt_lanes])
        lane_markings = lane_markings.astype('float').reshape(-1, 2)
        sample = {'image': image, 'lane_markings': lane_markings}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def display_sample(self, idx):
        img = self.__getitem__(idx)
        plt.imshow(img)

if __name__ == '__main__':
    json = r'C:\Users\user\Desktop\Personal\Final Year Project\TuSimple Dataset\Train Set\label_data_0601.json'
    root = r'C:\Users\user\Desktop\Personal\Final Year Project\TuSimple Dataset\Train Set'
    loader = TrueSimpleSet(json, root)
    print(loader.lane_markings.columns)
    loader.display_sample(1)
