import torch


def gen_loss(y, y_predict, adverserial_loss=0, alpha=1.0):
    cross_entropy_loss = pixel_wise_cross_entropy(y, y_predict)
    return cross_entropy_loss - alpha * adverserial_loss


def pixel_wise_cross_entropy(y, y_predict):
    y_predict = torch.clamp(y_predict, min=1e-6)
    w, h = y.shape[-2], y.shape[-1]
    cross_entropy = -torch.sum(torch.xlogy(y, y_predict)) / (w * h)
    print(cross_entropy)
    return torch.mean(cross_entropy)


def embedding_loss(fake_embedding, real_embedding):
    difference = fake_embedding - real_embedding
    return -torch.norm(difference, p=2)


def process_gen_output(gen_output):
    """Sets the class as the class with the highest probabilities"""
    return torch.round(gen_output)
