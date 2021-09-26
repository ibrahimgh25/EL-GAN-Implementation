r"""
This is to create simpler models I can experiment with on my local environment, which can't handle the 
 full models.
I basically copied the code from ..dense.el_gan.discriminator and ..dense.el_gan.genearator and reduced
 the number of dense blocks
"""
from torch import sigmoid
from itertools import zip_longest

from torch import cat
from torch.nn import Sequential
from torch.nn import ELU
import sys

from torch.nn.modules.activation import LeakyReLU
sys.path.append('..')
from dense import FCDenseNet
from dense.densenet import (
            FeatureBlock,
            Transition
)
from dense.shared import DenseBlock

class Generator(FCDenseNet):
    def __init__(self):
        super().__init__( 
                     in_channels = 3,
                     out_channels = 1,
                     initial_num_features = 25,
                     dropout = 0.1,

                    down_dense_growth_rates = 16,
                    down_dense_bottleneck_ratios = None,
                    down_dense_num_layers = (1, 2, 4, 6, 8),
                    down_transition_compression_factors = 0.5,

                    middle_dense_growth_rate = 16,
                    middle_dense_bottleneck = None,
                    middle_dense_num_layers = 8,

                    up_dense_growth_rates = 16,
                    up_dense_bottleneck_ratios = None,
                    up_dense_num_layers = (8, 6, 4, 2, 1),
                    )

    def forward(self, x):
        res = super().forward(x)
        return sigmoid(res)


class Discriminator(Sequential):
    def __init__( self,
                  in_channels=1,
                  initial_num_features=25,
                  dense_blocks_growth_rates=8,
                  dense_blocks_bottleneck_ratios=[None],
                  output_classes=2,
                  transition_blocks_compression_factors=0.5,
                  dropout=0):
        super().__init__()
        # region Parameters handling
        self.in_channels = in_channels
        self.output_classes = output_classes

        dense_block_args = (dropout,
                            dense_blocks_bottleneck_ratios, 
                            transition_blocks_compression_factors,
                            dense_blocks_growth_rates)
        self.markings_head, num_channels_1 = self._separtate_part(1, initial_num_features, *dense_block_args)
        self.full_img_head, num_channels_2 = self._separtate_part(3, initial_num_features, *dense_block_args)
        current_channels = num_channels_1 + num_channels_2
        self.common_part, current_channels = self._common_part(current_channels, *dense_block_args)
        
    def _separtate_part(self, input_channels,
                        initial_num_features, 
                        dropout, 
                        dense_blocks_bottleneck_ratios, 
                        transition_blocks_compression_factors,
                        dense_blocks_growth_rates):
        
        model = Sequential()
        # Initial Convolution
        features = FeatureBlock(input_channels, initial_num_features)
        current_channels = features.out_channels
        model.add_module('features', features)
        
        # Dense blocks
        name = f'seperate{input_channels}'
        num_layers = [1, 2]
        model, current_channels = self.add_demse_blocks(model, name,
                                    num_layers, current_channels,
                                    dropout, dense_blocks_bottleneck_ratios,
                                    transition_blocks_compression_factors,
                                    dense_blocks_growth_rates)
        # Return the current channels to be used by next part
        return model, current_channels

    def _common_part(self, 
                        initial_channels,
                        dropout,
                        dense_blocks_bottleneck_ratios,
                        transition_blocks_compression_factors,
                        dense_blocks_growth_rates):
        

        model = Sequential()
        num_layers = [3, 4, 6]
        # Add the dense blocks
        name  = 'common'
        model, current_channels = self.add_demse_blocks(model, name, 
                                    num_layers, initial_channels,
                                    dropout, dense_blocks_bottleneck_ratios,
                                    transition_blocks_compression_factors,
                                    dense_blocks_growth_rates)
        return model, current_channels

    def add_demse_blocks(self, model,
                name, num_layers,
                initial_channels, dropout,
                dense_blocks_bottleneck_ratios,
                transition_blocks_compression_factors,
                dense_blocks_growth_rates):
        
        num_blocks = len(num_layers)
        dense_blocks_bottleneck_ratios = self.numeric_to_tuple(dense_blocks_bottleneck_ratios, num_blocks)
        transition_blocks_compression_factors = self.numeric_to_tuple(transition_blocks_compression_factors, num_blocks)
        dense_blocks_growth_rates = self.numeric_to_tuple(dense_blocks_growth_rates, num_blocks)

        current_channels = initial_channels
        dense_blocks_params = [
            {
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br,
                    'nonlinearity':ELU
                }
            }
            for gr, nl, br in zip(dense_blocks_growth_rates, num_layers, dense_blocks_bottleneck_ratios)
        ]   
        transition_blocks_params = [
            {
                'compression': c
            }
            for c in transition_blocks_compression_factors
        ]

        block_pairs_params = zip_longest(dense_blocks_params, transition_blocks_params)
        for block_pair_idx, (dense_block_params, transition_block_params) in enumerate(block_pairs_params):
            block = DenseBlock(current_channels, **dense_block_params)
            current_channels = block.out_channels
            model.add_module(f'{name}_dense{block_pair_idx}', block)

            if transition_block_params is not None:
                transition = Transition(current_channels, **transition_block_params)
                current_channels = transition.out_channels
                model.add_module(f'{name}_trans{block_pair_idx}', transition)
        
        return model, current_channels

    def numeric_to_tuple(self, value, length):
        ''' Create a tuple with a single value'''
        if type(value) == int or type(value) == float:
            return (value, ) * length
        return value
    
    def forward(self, img, markings):
        # You first pass the label through the markings head
        label_part = self.markings_head(markings)
        # Then pass the image through the image head
        img_part = self.full_img_head(img)
        # Concatenate the two outputs into one tensor
        x_cat = cat((label_part, img_part), 1)
        embedding = self.common_part(x_cat)
        return embedding
    
    def apply(self, function):
        modules = [
                    self.markings_head,
                    self.full_img_head, 
                    self.common_part
                    ]
        for module in modules:
            module.apply(function)