import unittest
import torch
from src.el_gan.discriminator import (
    Discriminator,
    DiscriminatorParameters,
)


class TestDiscriminatorModel(unittest.TestCase):
    def test_init_noException(self):
        model = Discriminator(DiscriminatorParameters())

    def test_forward_expectedShape(self):
        model = Discriminator(DiscriminatorParameters())
        three_channel_image = torch.rand(2, 3, 512, 512)
        mask = torch.rand(2, 1, 512, 512)

        output = model(three_channel_image, mask)
        self.assertEquals(output[0].shape[0], 2)
        self.assertEquals(output[0].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
