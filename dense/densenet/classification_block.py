from torch.nn import Sequential, BatchNorm2d, ReLU, AvgPool2d, Linear

from ..shared import Flatten
from ..utils import RichRepr


class ClassificationBlock(RichRepr, Sequential):
    r"""
    Classification block for [DenseNet](https://arxiv.org/abs/1608.06993),
    takes in a feature map and outputs logit scores for classification
    """

    def __init__(self, in_channels, output_classes, linear_nodes=128, nonlinearity=ReLU):
        super(ClassificationBlock, self).__init__()

        self.in_channels = in_channels
        self.out_classes = output_classes

        self.add_module('norm', BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nonlinearity(inplace=True))
        self.add_module('pool', AvgPool2d(kernel_size=2, stride=1, padding=1))
        self.add_module('flatten', Flatten())
        self.add_module('linear', Linear(linear_nodes, output_classes))

    def __repr__(self):
        return super(ClassificationBlock, self).__repr__(self.in_channels, self.out_classes)
