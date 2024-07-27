from torch.nn.functional import softmax

from ..fc_densenet import FCDenseNet


class GeneratorParams:
    INITIAL_FEATURES = 75
    NUM_LAYERS_DOWN = [1, 2, 3, 4, 5, 6, 8]
    NUM_LAYERS_MIDDLE = 8
    GROWTH_RATE = 18
    COMPRESSION_FACTOR = 0.5
    DROPOUT = 0.1


class Generator(FCDenseNet):
    def __init__(self, config: GeneratorParams):
        super().__init__(
            in_channels=3,
            out_channels=1,
            initial_num_features=config.INITIAL_FEATURES,
            dropout=config.DROPOUT,
            down_dense_growth_rates=config.GROWTH_RATE,
            down_dense_num_layers=config.NUM_LAYERS_DOWN,
            down_transition_compression_factors=config.COMPRESSION_FACTOR,
            middle_dense_growth_rate=config.GROWTH_RATE,
            middle_dense_num_layers=config.GROWTH_RATE,
            up_dense_growth_rates=config.GROWTH_RATE,
            up_dense_num_layers=reversed(config.NUM_LAYERS_DOWN),
        )

    def forward(self, x):
        res = super().forward(x)
        return softmax(res, dim=1)
