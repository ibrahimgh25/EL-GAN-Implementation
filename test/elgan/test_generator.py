import unittest
import torch
from src.el_gan.generator import (
    Generator,
    GeneratorParams,
)


class TestDiscriminatorModel(unittest.TestCase):
    def test_init_noException(self):
        model = Generator(GeneratorParams())

    def test_forward_expectedShape(self):
        model = Generator(GeneratorParams())
        three_channel_image = torch.rand(2, 3, 512, 512)

        output = model(three_channel_image)
        self.assertEquals(output.shape, (2, 1, 512, 512))


if __name__ == "__main__":
    unittest.main()
