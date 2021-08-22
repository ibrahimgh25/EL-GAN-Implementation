from torch import save, no_grad
from torch.nn import Sequential

class Trainer(Sequential):
    def __init__(self, model,
                 loss_criterion,
                 optimizer, optimizer_params,
                 lr_scheduler=None, lr_scheduler_params={}):
        super().__init__()
        self.model = model
        self.criterion = loss_criterion
        self.optimizer = optimizer(params=model.parameters(), **optimizer_params)
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler(optimizer=self.optimizer, **lr_scheduler_params)

    def forward(self, *inputs_):
        return self.model.forward(*inputs_)
    
    def test(self, *inputs_):
        with no_grad():
            output = self.model.forward(*inputs_)
        return output
    
    def get_loss(self, y, y_predicted, *args, **kwargs):
        loss = self.criterion(y, y_predicted, *args, **kwargs)
        return loss.detach()
    
    def backwards(self, y, y_predicted, update_weights=True, *args, **kwargs):
        loss = self.criterion(y, y_predicted, *args, **kwargs)
        if update_weights:
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss.detach()
    
    def save_model_img(self, save_path):
        save(self.model, save_path)