from torch.nn.functional import softmax
from ..fc_densenet import FCDenseNet

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
        return softmax(res, dim=1)