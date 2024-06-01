# This file contains code adapted from pytorch-densenet-tiramisu by Federico Baldassarre
# Original Source: https://github.com/baldassarreFe/pytorch-densenet-tiramisu
# License: MIT License (https://opensource.org/licenses/MIT)


from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU

from ..utils import RichRepr


class Bottleneck(RichRepr, Sequential):
    r"""
    A 1x1 convolutional layer, followed by Batch Normalization and ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module(
            "conv", Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.add_module("norm", BatchNorm2d(num_features=out_channels))
        self.add_module("relu", ReLU(inplace=True))

    def __repr__(self):
        return super(Bottleneck, self).__repr__(self.in_channels, self.out_channels)
