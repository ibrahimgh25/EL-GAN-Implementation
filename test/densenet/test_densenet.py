# This file contains code adapted from pytorch-densenet-tiramisu by Federico Baldassarre
# Original Source: https://github.com/baldassarreFe/pytorch-densenet-tiramisu
# License: MIT License (https://opensource.org/licenses/MIT)

import unittest

import torch

from src.densenet import DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201
from src.utils import count_parameters, count_conv2d


class TestDenseNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.rgb_channels = 3
        imagenet_h = 224
        imagenet_w = 224
        self.imagenet_classes = 1000

        self.images = torch.empty(
            self.batch_size, self.rgb_channels, imagenet_h, imagenet_w
        )
        print("Images:", self.images.shape)

    def test_densenet(self):
        densenets = [
            DenseNet(),
            DenseNet121(),
            DenseNet161(),
            DenseNet169(),
            DenseNet201(),
        ]

        for densenet in densenets:
            with self.subTest(klass=type(densenet).__name__):
                print(densenet)
                layers = count_conv2d(densenet)
                print("Layers:", layers)
                print("Parameters:", count_parameters(densenet))

                logits = densenet(self.images)
                print("Logits:", logits.shape)
                self.assertEqual(
                    logits.shape, torch.Size((self.batch_size, self.imagenet_classes))
                )


if __name__ == "__main__":
    unittest.main()
