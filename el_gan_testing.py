from dense.shared.dense_block import DenseBlock
import sys, os
from torch.optim import Adam, SGD
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
from dense.el_gan import Generator, Discriminator
from training_utils import *

if __name__ =='__main__':
    json = r'label_data_mini.json'
    root_dir = r'C:\Users\user\Desktop\Mini Dataset'
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = LaneDataSet(json, root_dir)
    train_gen = DataLoader(train_set, batch_size=1, shuffle=True)
    optimizer_params = {'lr':5e-3}    
    lr_scheduler_params = {'gamma':0.99}

    disc_trainer = Trainer(Discriminator(), embedding_loss,
                            SGD, optimizer_params,
                            ExponentialLR, lr_scheduler_params)
    
    fake_embedding = None
    for index, data in enumerate(train_gen):
        inputs, labels = data
        # Loading a datasample might fail at some point,
        # if that happens, I just skip the sample
        if torch.all(torch.eq(labels, torch.tensor(1))):
            continue
        
        real_score, real_embedding = disc_trainer(inputs.float(), labels.float())
        print(real_embedding.size(), 'real_embedding_baby')
        if fake_embedding is None:
            fake_embedding = disguise_label(real_embedding)
            continue
        else:
            loss = disc_trainer.backwards(real_embedding, real_embedding)
            break