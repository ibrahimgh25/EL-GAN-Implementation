# This module is for pretraining the generator, all what I did is copy the parts from the 
# elgan_training.py file and remove all parts related to the discriminator
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

# from dense.el_gan import Generator, Discriminator
from training_utils import *
from training_utils.simplified_models import Generator

########################################################################################
######################## Defining and Initializing the Networks ########################
########################################################################################
def initialize(m, nonlinearity='relu'):
  ''' Just initializaes a model with kaiming_normal'''
  try:
    if len(m.weight.size()) > 2:
      torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
  except:
    pass
  return m

# If there's a GPU we're going to use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the generator and discriminator, then move them to 'device'
generator = Generator().to(device)

# Initialize the generator
generator = generator.apply(initialize)
# Now we create the trainer in an effort to make our code a little less messy
adam_params = {'lr':5e-4, 'betas':(0.9, 0.99), 'weight_decay':3e-4}
sgd_params = {'lr':1e-5, 'weight_decay':3e-4}
scheduler_params = {'gamma':0.99}
gen_trainer = Trainer(generator, gen_loss, 
                      Adam, adam_params,
                      ExponentialLR, scheduler_params)
################################# End Region ###########################################

########################################################################################
######################## Defining the Dataloaders  #####################################
########################################################################################
# The json is the file containing labels for the TrueSimple dataset
training_json = r'label_data_mini.json'
testing_json = r'label_data_mini.json'
# The root directory is the directory where the dataset is located
test_dir = r'C:\Users\user\Desktop\Mini Dataset'
train_dir = r'C:\Users\user\Desktop\Mini Dataset'

train_set = LaneDataSet(training_json, train_dir)
test_set = LaneDataSet(testing_json, test_dir)
params = {'batch_size': 1,
        'shuffle': True}
# Create the dataloaders from the defined datasets
train_gen_loader = DataLoader(train_set, **params)
test_gen_loader = DataLoader(test_set, **params)
################################# End Region ###########################################

########################################################################################
########################### Training Region  ###########################################
########################################################################################
# Prepare tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/elgan')



EPOCHS = 4

for epoch in range(EPOCHS):
    # Define the running losses for the generator and the disc
    gen_running_loss = 0.0

    # Training the network
    for i, data in enumerate(train_gen_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        # Loading a datasample might fail at some point,
        # if that happens, I just skip the sample
        if torch.all(torch.eq(labels, torch.tensor(1))):
            continue
        # Get the generator output
        gen_output = gen_trainer(inputs)
        # Update the generator if now's its turn
        gen_loss = gen_trainer.backwards(labels, gen_output, i % 500 < 300)
        # Detach the output from the generator

        # Update the running losses
        gen_running_loss += gen_loss.item()
        # Save the data and reset running losses every 100 iterations
        if not i % 100 and i != 0:
          gen_avg = gen_running_loss / 100
          print(f"Training -- Epoch {epoch}, iteration {i}: gen_loss: {gen_avg}")
          current_iter = epoch * len(train_gen_loader) + i
          writer.add_scalar('Generator Training Loss', gen_avg, current_iter)
          gen_running_loss = 0.0
    
    # Save the models after each epoch
    gen_trainer.save(f'models/generator/generator_epoch_{epoch}.ptmdl')
    # Testing the networks
    gen_running_loss = 0.0
    for i, data in enumerate(test_gen_loader):
      # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        # Loading a datasample might fail at some point,
        # if that happens, I just skip the sample
        if torch.all(torch.eq(labels, torch.tensor(1))):
            continue
        # Get the generator output
        gen_output = gen_trainer.test(inputs)
        # Update the generator if now's its turn
        gen_loss = gen_trainer.get_loss(labels, gen_output)

        # Update the running loss
        gen_running_loss += gen_loss.item()
        # Save the data and reset running losses every 100 iterations
        if not i % 100 and i != 0:
          gen_avg = gen_running_loss / 100
          print(f"Testing -- Epoch {epoch}, iteration {i}: gen_loss: {gen_avg}")
          current_iter = epoch * len(train_gen_loader) + i
          writer.add_scalar('Generator Training Loss', gen_avg, current_iter)
          gen_running_loss = 0.0

writer.exit()
