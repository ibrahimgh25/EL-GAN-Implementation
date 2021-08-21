from dense.shared.dense_block import DenseBlock
import sys, os
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from dense.el_gan.elgan import Generator, pixel_cce
from dense.el_gan.dataloader import LaneDataSet
from dense.el_gan.elgan import Discriminator
import cv2 as cv
if __name__ =='__main__':
    json = r'label_data_mini.json'
    root_dir = r'C:\Users\user\Desktop\Mini Dataset'
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = LaneDataSet(json, root_dir)
    params = {'batch_size': 1,
            'shuffle': True}
    train_gen = DataLoader(train_set, **params)
    gen = Generator()
    for index, data in enumerate(train_gen):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if torch.all(torch.eq(labels, torch.tensor(1))):
            continue
        output = gen(inputs.float())
        break