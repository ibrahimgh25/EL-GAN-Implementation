import torch
from torch.types import Number

from torch.nn.functional import softmax
from ..fc_densenet import FCDenseNet
from ..densenet import FeatureBlock, ClassificationBlock, Transition
from ..shared import DenseBlock

from itertools import zip_longest

from torch.nn import Sequential
from torch.nn import functional as F

from ..shared import DenseBlock

class Generator(FCDenseNet):
    def __init__(self):
        super().__init__( in_channels = 3,
                     out_channels = 2,
                     initial_num_features = 75,
                     dropout = 0.1,

                    down_dense_growth_rates = 18,
                    down_dense_bottleneck_ratios = None,
                    down_dense_num_layers = (1, 2, 3, 4, 5, 6, 8),
                    down_transition_compression_factors = 0.8,

                    middle_dense_growth_rate = 18,
                    middle_dense_bottleneck = None,
                    middle_dense_num_layers = 8,

                    up_dense_growth_rates = 18,
                    up_dense_bottleneck_ratios = None,
                    up_dense_num_layers = (8, 6, 5, 4, 3, 2, 1))

    def forward(self, x):
        res = super().forward(x)
        return softmax(res)

class Discriminator(Sequential):
    def __init__( self,
                  in_channels=2,
                  initial_num_features=24,
                  dense_blocks_growth_rates=8,
                  dense_blocks_bottleneck_ratios=4,
                  output_classes=2,
                  transition_blocks_compression_factors=0.8,
                  dropout=0):
        super().__init__()
        # region Parameters handling
        self.in_channels = in_channels
        self.output_classes = output_classes

        dense_block_args = (dropout,
                            dense_blocks_bottleneck_ratios, 
                            transition_blocks_compression_factors,
                            dense_blocks_growth_rates)
        self.markings_head, num_channels_1 = self._separtate_part(2, initial_num_features, *dense_block_args)
        self.full_img_head, num_channels_2 = self._separtate_part(3, initial_num_features, *dense_block_args)
        current_channels = num_channels_1 + num_channels_2
        self.common_part, current_channels = self._common_part(current_channels, *dense_block_args, 2)
        self.classification_block = ClassificationBlock(current_channels, output_classes, 936)
    
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
        # Return the current channels to be used by nex part
        return model, current_channels

    def _common_part(self, 
                        initial_channels,
                        dropout,
                        dense_blocks_bottleneck_ratios,
                        transition_blocks_compression_factors,
                        dense_blocks_growth_rates,
                        output_classes):
        

        model = Sequential()
        num_layers = [3, 4, 6, 8, 8]
        # Add the dense blocks
        name  = 'common'
        model, current_channels = self.add_demse_blocks(model, name, 
                                    num_layers, initial_channels,
                                    dropout, dense_blocks_bottleneck_ratios,
                                    transition_blocks_compression_factors,
                                    dense_blocks_growth_rates)
        # region Classification block
        # model.add_module('classification', ClassificationBlock(current_channels, output_classes, 936))
        # endregion

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
                    'bottleneck_ratio': br
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
        label_part = self.markings_head(markings)
        img_part = self.full_img_head(img)
        x_cat = torch.cat((label_part, img_part), 1)
        # Embedding to be used to calculate the embedding loss
        embedding = self.common_part(x_cat)
        y = self.classification_block(embedding)
        return softmax(y, dim=1), embedding

def pixel_cce(y, y_predict):
    ''' An implementation for pixel-wise cross entropy'''
    y_predict = torch.add(y_predict, torch.tensor(1e-15))
    w = torch.tensor(y.shape[-2])
    h = torch.tensor(y.shape[-1])
    loss = -torch.sum(torch.xlogy(y, y_predict))
    loss = torch.divide(loss, torch.multiply(w, h))
    return loss

def embedding_loss(fake_embedding, real_embedding):
    ''' Basically euclidean distance'''
    return torch.cdist(fake_embedding, real_embedding)

def disguise_label(label, low_end=0, high_end=0.1):
    ''' Can be used to subtract and add random values to a label so it isn't easily spotted by a discriminator'''
    noise = torch.FloatTensor(*label.size()).uniform_(low_end, high_end)
    mask = label
    mask[mask==0] = -1
    return torch.subtract(label, torch.multiply(noise, mask))