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
BATCH_SIZE    = 1 # batch size
MODEL_PATH    = 'models/checkpoints/test.pth' # where to locate the data
LOG_PATH      = 'logs/test.log' # where to save the log

def extract_feature(device, model, data_loader, data_size):
    model.eval()

    extracted_features = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            out, extracted_feature = model(data)
        extracted_features.append(extracted_feature)
    
    return extracted_features


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

with open('data/test_dataset_0227.pkl', 'rb') as f:
    dataset = pickle.load(f)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

if torch.cuda.is_available():
    logging.info('Using GPU')
else:
    logging.info('Using CPU')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCNNet().to(device)
try:
    loaded_model_state = torch.load(MODEL_PATH)
    model.load_state_dict(loaded_model_state['state_dict'])
except:
    raise Exception("Model state not found.")

## checking final test results
extracted_features = extract_feature(device, model, loader, len(dataset))
print(len(extracted_features))
print(extracted_features[0])