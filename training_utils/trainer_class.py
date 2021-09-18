import _pickle as cpickle
import matplotlib.pyplot as plt
import numpy as np

from torch import no_grad
from torch.nn import Sequential

class Trainer(Sequential):
    def __init__(self, model,
                 loss_criterion,
                 optimizer, optimizer_params,
                 lr_scheduler=None, lr_scheduler_params={},
                 lr_scheduling_period=200):
        super().__init__()
        self.model = model
        self.criterion = loss_criterion
        self.optimizer = optimizer(params=model.parameters(), **optimizer_params)
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler(optimizer=self.optimizer, **lr_scheduler_params)
        else:
            self.lr_scheduler = None
        self.iters = 0
        self.lr_scheduling_period = lr_scheduling_period
        self.training_losses = []
        self.testing_losses = []
        
    def forward(self, *inputs_):
        return self.model.forward(*inputs_)
    
    def test(self, *inputs_):
        with no_grad():
            output = self.model.forward(*inputs_)
        return output
    
    def get_loss(self, y, y_predicted, *args, **kwargs):
        loss = self.criterion(y, y_predicted, *args, **kwargs)
        return loss.detach()
    
    def backwards(self, y, y_predicted, update_weights=True, update_lr=False, *args, **kwargs):
        loss = self.criterion(y, y_predicted, *args, **kwargs)
        if update_weights:
            loss.backward()
            self.optimizer.step()
            self.iters += 1
        # If we need to update the learning rate, we do so
        if self.iters == self.lr_scheduling_period:
            self.lr_scheduler.step()
            self.iters = 0
        # We zero the gradient anyway to not carry the gradient to future iterations
        self.optimizer.zero_grad()
        return loss.detach()
    
    def save_model_img(self, save_path):
        with open(save_path, 'wb') as file_out:
            cpickle.dump(self, file_out)
    
    def load_model_img(self, filename):
        with open(filename, 'rb') as file_in:
            cpickle.load(file_in)
    
    def load_model_only(self, filename):
        with open(filename, 'rb') as file_in:
            trainer = cpickle.load(file_in)
        self.model = trainer.model
    
    def log_training_loss(self, iteration, value):
        self.training_losses.append((iteration, value))
    
    def log_testing_loss(self, iteration, value):
        self.testing_losses.append((iteration, value))
    
    def show_img(self, model_output):
        img = model_output.detach().cpu().numpy()[0]
        img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        plt.imshow(img * 255)
        plt.show()
