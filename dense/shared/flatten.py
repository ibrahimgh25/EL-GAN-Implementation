# This file contains code adapted from pytorch-densenet-tiramisu by Federico Baldassarre
# Original Source: https://github.com/baldassarreFe/pytorch-densenet-tiramisu
# License: MIT License (https://opensource.org/licenses/MIT)


from torch.nn import Module


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
