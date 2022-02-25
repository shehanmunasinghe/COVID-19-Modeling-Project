import torch

from utils.data import *
from utils.visualize import *
from utils.trainer import *
from utils.forward_model import SIRDModel
from utils.nn_model import *
from utils.loss import *

import yaml
import argparse
import random
import numpy as np



##################################################
# Config
##################################################
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument("config", help="train config file path")
args = parser.parse_args()

with open(args.config) as conf_file:
    conf = yaml.full_load(conf_file)
print("Using configuration: ", conf)

# Random Seeds
np.random.seed(conf['SEEDS']['SEED_NUMPY'])
torch.manual_seed(conf['SEEDS']['SEED_TORCH'])
random.seed(conf['SEEDS']['SEED_PYRANDOM'])

##################################################
# Get Real Data
##################################################

# build dataset
population,pdnf, trainset = build_dataset_from_config(conf)
# Plot Real Data
# plot_real_data(trainset)


##################################################
# Building the model 
##################################################

num_days_iter = len(trainset) - conf["MODEL"]["PAST_N_DAYS"]

# build NN model
nn_model = build_nn_model_from_config(conf)

# build forward model
forward_model = SIRDModel(nn_model = nn_model, population = population )


optimizer = build_optimizer_from_config(conf, forward_model)

criterion = build_loss_from_config(conf, population_denormalization_factor=pdnf)

##################################################
# Fitting the model 
##################################################

try:    
    do_training(
        optimizer,
        num_days_iter, 
        trainset, 
        conf["MODEL"]["PAST_N_DAYS"],
        forward_model,
        criterion,
        epochs=conf['OPTIMIZATION']['NUM_EPOCHS']
    )
except KeyboardInterrupt:
        print('Interrupted')


##################################################
### Visualize predictions of the trained model
##################################################

do_inference(
    forward_model, 
    trainset,
    num_days_iter,
    conf["MODEL"]["PAST_N_DAYS"]
)

# Show plots
plt.show()