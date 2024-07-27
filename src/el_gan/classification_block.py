import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationBlock(nn.Sequential):
    def __init__(self, in_channels, out_classes, dropout_rate=0.5):
        super(ClassificationBlock, self).__init__(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=10),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=10),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_classes),
            nn.Softmax(dim=1),
        )
