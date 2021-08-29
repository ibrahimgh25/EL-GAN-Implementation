import sys, os
import torch
from torch.optim import Adam, SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

# from dense.el_gan import Generator, Discriminator
from training_utils import *
from training_utils.simplified_models import Discriminator, Generator

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

# Define the generator and discriminator, then move them to 'device'
generator = Generator().to(device)
disc = Discriminator().to(device)

# Initialize each part of the network, this is a little messy way to implement it
# but I've done non the less since it helps to see the initialization phase
# Disc is technichally a class, not a model, it's a bunch of models sewed together
parts_to_initialize = [
              generator, disc.markings_head,
              disc.full_img_head, 
              disc.common_part,
              disc.classification_block
]
for module in parts_to_initialize:
  modeul = module.apply(initialize)
# Now we create the trainer in an effort to make our code a little less messy
adam_params = {'lr':5e-4, 'betas':(0.9, 0.99), 'weight_decay':3e-4}
sgd_params = {'lr':1e-5, 'weight_decay':3e-4}
scheduler_params = {'gamma':0.99}
gen_trainer = Trainer(generator, gen_loss, 
                      Adam, adam_params,
                      ExponentialLR, scheduler_params)

disc_trainer = Trainer(disc, embedding_loss,
                      SGD, sgd_params,
                      ExponentialLR, scheduler_params)
################################# End Region ###########################################

########################################################################################
######################## Defining the Dataloaders  #####################################
########################################################################################
# The json is the file containing labels for the TrueSimple dataset
training_json = r'label_data_mini.json'
testing_json = r'label_data_mini.json'
# The root directory is the directory where the dataset is located
root_dir = r'C:\Users\user\Desktop\Mini Dataset'

train_set = LaneDataSet(training_json, root_dir)
test_set = LaneDataSet(testing_json, root_dir)
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
    gen_running_loss, disc_running_loss = 0.0, 0.0

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
        gen_output = gen_output.detach()
        # Run the the discriminator on the fake and real markings
        real_embedding = disc_trainer(inputs, labels)
        fake_embedding = disc_trainer(inputs, gen_output)
        # Update the discriminator if now's its turn
        disc_loss = disc_trainer.backwards(real_embedding, fake_embedding, i % 500 >= 300)

        # Update the running losses
        gen_running_loss += gen_loss.item()
        disc_running_loss += disc_loss.item()
        # Save the data and reset running losses every 100 iterations
        if not i % 100 and i != 0:
          gen_avg = gen_running_loss / 100
          disc_avg = disc_running_loss / 100
          print(f"Training -- Epoch {epoch}, iteration {i}: gen_loss: {gen_avg}, disc_loss: {disc_avg}")
          current_iter = epoch * len(train_gen_loader) + i
          writer.add_scalar('Generator Training Loss', gen_avg, current_iter)
          writer.add_scalar('Discriminator Training Loss', disc_avg, current_iter)
          gen_running_loss, disc_running_loss = 0, 0
    
    # Save the models after each epoch
    gen_trainer.save(f'models/generator/generator_epoch_{epoch}.ptmdl')
    disc_trainer.save(f'models/discriminator/discriminator_epoch_{epoch}.ptmdl')
    # Testing the networks
    gen_running_loss, disc_running_loss = 0.0, 0.0
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
        # Detach the output from the generator
        gen_output = gen_output.detach()
        # Run the the discriminator on the fake and real markings
        real_embedding = disc_trainer.test(inputs, labels)
        fake_embedding = disc_trainer.test(inputs, gen_output)
        # Update the discriminator if now's its turn
        disc_loss = disc_trainer.get_loss(real_embedding, fake_embedding)

        # Update the running losses
        gen_running_loss += gen_loss.item()
        disc_running_loss += disc_loss.item()
        # Save the data and reset running losses every 100 iterations
        if not i % 100 and i != 0:
          gen_avg = gen_running_loss / 100
          disc_avg = disc_running_loss / 100
          print(f"Testing -- Epoch {epoch}, iteration {i}: gen_loss: {gen_avg}, disc_loss: {disc_avg}")
          current_iter = epoch * len(train_gen_loader) + i
          writer.add_scalar('Generator Training Loss', gen_avg, current_iter)
          writer.add_scalar('Discriminator Training Loss', disc_avg, current_iter)
          gen_running_loss, disc_running_loss = 0, 0
     

writer.exit()
