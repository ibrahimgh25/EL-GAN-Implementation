from itertools import zip_longest

from torch import cat
import torch
from torch.nn import Sequential
from torch.nn.functional import softmax
from torch.nn.modules.activation import ELU

from src.densenet.transition import Transition

from ..densenet import FeatureBlock
from ..shared import DenseBlock
from .classification_block import ClassificationBlock


class DiscriminatorParameters:
    INITIAL_FEATURES = 24
    GROWTH_RATE = 8
    COMPRESSION_FACTOR = 0.5
    DROPOUT = 0
    BOTTLE_NECK_RATIO = 4
    SEPARATE_LAYERS_NUM = [1, 2]
    COMMON_LAYERS_NUM = [3, 4, 6, 8, 8]


class Discriminator(Sequential):
    def __init__(self, config: DiscriminatorParameters):
        super().__init__()
        self.config = config

        self.markings_head = self._separate_part(1)
        self.full_img_head = self._separate_part(3)

        current_channels = (
            self.markings_head[-1].out_channels + self.full_img_head[-1].out_channels
        )
        self.common_part = self._common_part(current_channels)

        self.classification_block = ClassificationBlock(
            self.common_part[-1].out_channels, 2
        )

    def _separate_part(self, input_channels):
        model = Sequential(FeatureBlock(input_channels, self.config.INITIAL_FEATURES))
        model = self.add_dense_blocks(
            model,
            name=f"Separate_{input_channels}_",
            num_layers=self.config.SEPARATE_LAYERS_NUM,
            initial_channels=self.config.INITIAL_FEATURES,
        )
        return model

    def _common_part(self, initial_channels):
        model = self.add_dense_blocks(
            Sequential(),
            name="common",
            num_layers=self.config.COMMON_LAYERS_NUM,
            initial_channels=initial_channels,
        )
        return model

    def add_dense_blocks(self, model, name, num_layers, initial_channels):
        current_channels = initial_channels
        for idx, conv_num in enumerate(num_layers):
            dense_block_params = {
                "growth_rate": self.config.GROWTH_RATE,
                "num_layers": conv_num,
                "dense_layer_params": {
                    "dropout": self.config.DROPOUT,
                    "bottleneck_ratio": self.config.BOTTLE_NECK_RATIO,
                    "nonlinearity": ELU,
                },
            }
            block = DenseBlock(current_channels, **dense_block_params)
            current_channels = block.out_channels
            model.add_module(f"{name}_dense{idx}", block)

            transition = Transition(
                current_channels, compression=self.config.COMPRESSION_FACTOR
            )
            current_channels = transition.out_channels
            model.add_module(f"{name}_trans{idx}", transition)
        return model

    def forward(self, img, markings):
        label_features: torch.Tensor = self.markings_head(markings)
        img_features = self.full_img_head(img)
        concatenated_features = cat((label_features, img_features), 1)
        embedding = self.common_part(concatenated_features)
        y = self.classification_block(embedding)
        return softmax(y, dim=1), embedding

    def apply(self, function):
        modules = [
            self.markings_head,
            self.full_img_head,
            self.common_part,
            self.classification_block,
        ]
        for module in modules:
            module.apply(function)
