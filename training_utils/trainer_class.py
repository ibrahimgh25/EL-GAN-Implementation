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

    def forward(self, *input_):
        return self.model.forward(*input_)
    
    def backwards(self, y, y_predicted, update_weights=True):
        # y = y.detach()
        loss = self.criterion(y, y_predicted)
        print(loss.size())
        if update_weights:
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss.detach()