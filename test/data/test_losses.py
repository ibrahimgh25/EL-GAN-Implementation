import unittest
import torch

from src.data.losses import (
    pixel_wise_cross_entropy,
    gen_loss,
    embedding_loss,
    process_gen_output,
)


class LossFunctionTestCase(unittest.TestCase):

    def test_pixel_wise_cross_entropy(self):
        y = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        y_predict = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        expected_loss = -0.5 * (
            torch.log(torch.tensor(0.9 + 1e-6)) + torch.log(torch.tensor(0.8 + 1e-6))
        )
        result = pixel_wise_cross_entropy(y, y_predict)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=5)

    def test_gen_loss(self):
        y = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        y_predict = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        adverserial_loss = torch.tensor(0.3)
        alpha = 1.0
        base_loss = pixel_wise_cross_entropy(y, y_predict)
        expected_loss = base_loss - alpha * adverserial_loss
        result = gen_loss(y, y_predict, adverserial_loss, alpha)
        self.assertAlmostEqual(result.item(), expected_loss.item(), places=5)

    def test_embedding_loss(self):
        real_embedding = torch.tensor([1.0, 2.0, 3.0])
        fake_embedding = torch.tensor([1.0, 2.0, 3.0])
        result = embedding_loss(real_embedding, fake_embedding)
        self.assertEqual(result.item(), 0)

    def test_process_gen_output(self):
        gen_output = torch.tensor([0.3, 0.6, 0.49, 0.51])
        expected_output = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = process_gen_output(gen_output)
        self.assertTrue(torch.equal(result, expected_output))


# Running the tests
if __name__ == "__main__":
    unittest.main()
