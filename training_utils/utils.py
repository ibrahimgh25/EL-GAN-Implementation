import torch

def gen_criterion(y, y_predict, adverserial_loss=0, alpha=1):
    ''' An implementation for pixel-wise cross entropy'''
    pred_inv = torch.ones_like(y_predict) - y_predict
    y_predict = torch.add(y_predict, torch.tensor(1e-15))
    pred_inv = torch.add(pred_inv, torch.tensor(1e-15))
    y_inv = torch.ones_like(y) - y
    w = torch.tensor(y.shape[-2])
    h = torch.tensor(y.shape[-1])
    loss = -torch.sum(torch.xlogy(y, y_predict) + torch.xlogy(y_inv, pred_inv))
    loss = torch.divide(loss, torch.multiply(w, h))
    return loss - alpha * adverserial_loss

def embedding_loss(fake_embedding, real_embedding):
    ''' Basically euclidean distance'''
    # The only way I could get this to work was this wierd way, I scoured the internet and found no 
    # solutions, but after expermentations I was able to solve the many errors I encountered
    # This may slow down training, I wil look into something else later
    dist = torch.tensor([-torch.sum(torch.sub(x, y)**2) for x, y in zip(real_embedding, fake_embedding)])
    dist.requires_grad = True
    return dist.sum()

def disguise_label(label, low_end=0, high_end=0.1, device='cuda:0'):
    ''' Can be used to subtract and add random values to a label so it isn't easily spotted by a discriminator'''
    noise = torch.FloatTensor(*label.size()).uniform_(low_end, high_end).to(device)
    mask = torch.clone(label)
    mask[mask==0] = -1
    return torch.subtract(label, torch.multiply(noise, mask))

def process_gen_output(gen_output):
    ''' Sets the class as the class with the highest probabilities'''
    gen_output[gen_output > 0.5] = 1
    gen_output[gen_output <= 0.5] = 0
    return gen_output

def gen_loss(y, y_predict, adverserial_loss=0, alpha=1.0):
    ''' An implementation for pixel-wise cross entropy'''
    y_predict = torch.add(y_predict, torch.tensor(1e-6))
    w = torch.tensor(y.shape[-2])
    h = torch.tensor(y.shape[-1])
    loss = -torch.sum(torch.xlogy(y, y_predict))
    loss = torch.divide(loss, torch.multiply(w, h))
    return loss - alpha * adverserial_loss
