import time
import logging
import argparse
import random
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops

from utils.loader import *
from models import *

## Hyper-parameter setting
SEED          = 1 # seed for random state
DATA_PATH     = '' # where to locate the data
LOG_PATH      = 'logs/test.log' # where to save the log
BATCH_SIZE    = 10 # batch size of data loader
LEARNING_RATE = 1e-1 # initial learning rate
LR_STEP_SIZE  = 20 # epochs before each lr decay
LR_DECAY      = 0.4 # multiplied by for lr decay
NUM_EPOCHS    = 100 # number of epochs for training

def train(device, model, optimizer, data_loader, data_size):
    model.train()

    total_loss = 0
    correct = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

        correct += out.max(dim=1)[1].eq(data.y).sum().item()

    train_loss = total_loss / data_size
    train_acc = correct / data_size
    
    return train_loss, train_acc

def test(device, model, data_loader, data_size):
    model.eval()

    total_loss = 0
    correct = 0
    predicted_y = []
    original_y = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = F.nll_loss(out, data.y)
        total_loss += loss.item() * data.num_graphs
        predicted_y.extend(out.max(dim=1)[1])
        original_y.extend(data.y)
        correct += out.max(dim=1)[1].eq(data.y).sum().item()

    test_loss = total_loss / data_size
    test_acc = correct / data_size
    test_out = (predicted_y, original_y)
    
    return test_loss, test_acc, test_out


## Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
logging.basicConfig(level=logging.ERROR, format='[%(asctime)s %(levelname)s] %(message)s')
logger = logging.getLogger()
hdlr = logging.FileHandler(LOG_PATH)
# hdlr = logging.FileHandler('logs/train_val_' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.log')
hdlr.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
logger.addHandler(hdlr)

## Ensure reproducibility, refering to https://blog.csdn.net/hyk_1996/article/details/84307108
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Create dataset with multiple data
# train_dataset, test_dataset = fromTxt2DatasetWithFeature(DATA_PATH + 'data/test_dpabi/', DATA_PATH + 'data/RANIAC_181210_345_sfMRI_90.csv')
# with open('data/train_dataset_0227.pkl', 'wb') as f:
#     pickle.dump(train_dataset, f)
# with open('data/test_dataset_0227.pkl', 'wb') as f:
#     pickle.dump(test_dataset, f)

with open('data/train_dataset_0227.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('data/test_dataset_0227.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if torch.cuda.is_available():
    logging.info('Using GPU')
else:
    logging.info('Using CPU')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = Net_191225().to(device)
model = GIN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## learning-rate scheduler.
scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY)

train_epochs = []
for epoch in range(1, NUM_EPOCHS+1):
    scheduler.step()
    train_loss, train_acc = train(device, model, optimizer, train_loader, len(train_dataset))
    test_loss, test_acc, _ = test(device, model, test_loader, len(test_dataset))
    epoch_res = 'Epoch {:03d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, train_loss, train_acc, test_loss, test_acc)
    logging.info(epoch_res)
    train_epochs.append(epoch_res)

## checking final test results
test_loss, test_acc, test_out = test(device, model, test_loader, len(test_dataset))
test_check = []
for idx in range(len(test_out[0])):
    test_out[0][idx] = test_out[0][idx].item()
    test_out[1][idx] = test_out[1][idx].item()
    if test_out[0][idx] == test_out[1][idx]:
        test_check.append(1)
    else:
        test_check.append(0)
print(test_out[0])
print(test_out[1])
print(test_check)