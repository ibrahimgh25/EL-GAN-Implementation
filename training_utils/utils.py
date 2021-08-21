import torch

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
    # The only way I could get this to work was this wierd way, I scoured the internet and found no 
    # solutions, but after expermentations I was able to solve the many errors I encountered
    # This may slow down training, I wil look into something else later
    cdist = lambda fake_embedding, real_embedding: -torch.sum(torch.cdist(fake_embedding, real_embedding))
    dist = torch.tensor([cdist(x, y) for x, y in zip(real_embedding, fake_embedding)])
    dist.requires_grad = True
    return dist

def disguise_label(label, low_end=0, high_end=0.1, device='cpu'):
    ''' Can be used to subtract and add random values to a label so it isn't easily spotted by a discriminator'''
    noise = torch.FloatTensor(*label.size(), device=device).uniform_(low_end, high_end)
    mask = torch.clone(label)
    mask[mask==0] = -1
    return torch.subtract(label, torch.multiply(noise, mask))