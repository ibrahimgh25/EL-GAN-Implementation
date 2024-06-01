# This file contains code adapted from pytorch-densenet-tiramisu by Federico Baldassarre
# Original Source: https://github.com/baldassarreFe/pytorch-densenet-tiramisu
# License: MIT License (https://opensource.org/licenses/MIT)


import unittest

import torch

from src.shared import DenseLayer
from src.utils import count_parameters


class TestDenseLayer(unittest.TestCase):

    def test_dense_layer(self):
        in_channels = 512
        out_channels = 8
        x = torch.empty(1, in_channels, 32, 64)
        print("x:", x.shape)

        layer_params = [
            {},
            {"dropout": 0.2},
            {"bottleneck_ratio": 4},
            {"bottleneck_ratio": 4, "dropout": 0.2},
        ]

        for params in layer_params:
            with self.subTest(**params):
                dense_layer = DenseLayer(in_channels, out_channels, **params)
                print(dense_layer)
                print("Parameters:", count_parameters(dense_layer))

                y = dense_layer(x)
                print("y:", y.shape)

                self.assertEqual(y.shape[1], out_channels)


if __name__ == "__main__":
    unittest.main()
