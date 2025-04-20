import os
import sys
import math
import random
import argparse
import pickle 
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchcde
from pytorch_metric_learning import losses

from src.config import *
from src.dataloader import *
from src.model import *
from src.trainer import *
from src.evaluation import *
from src.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
args = parse_args()
seed_everything(args.seed)

## Check if output already exists
if args.full_training:
    output_file = f'out_pretrain/{args.data_name}/{args.data_name}-full_{args.seed}'
else:
    output_file = f'out_pretrain/{args.data_name}/{args.data_name}_{args.seed}'

if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Skipping this run.")
    sys.exit(1)
    
#
args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

## Load data
with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    X_train_intp, X_train_shirink, X_train_forecast, y_train, X_val_intp, X_val_shirink, X_val_forecast, y_val, X_test_intp, X_test_shirink, X_test_forecast, y_test = pickle.load(f)

##
X_train_aug = X_train_intp
X_val_aug = X_val_intp
X_test_aug = X_test_intp

X_train_intp = torch.tensor(X_train_intp).transpose(1,2)
X_train_aug = torch.tensor(X_train_aug).transpose(1,2)
y_train = torch.tensor(y_train)

X_val_intp = torch.tensor(X_val_intp).transpose(1,2)
X_val_aug = torch.tensor(X_val_aug).transpose(1,2)
y_val = torch.tensor(y_val)

X_test_intp = torch.tensor(X_test_intp).transpose(1,2)
X_test_aug = torch.tensor(X_test_aug).transpose(1,2)
y_test = torch.tensor(y_test)

##
if args.full_training:
    X_train_intp = torch.cat([X_train_intp, X_val_intp, X_test_intp], dim=0)
    X_train_aug = torch.cat([X_train_aug, X_val_aug, X_test_aug], dim=0)
    y_train = torch.cat([y_train, y_val, y_test], dim=0)
    args.data_name = '-'.join([args.data_name, 'full'])

##
preprocessed_data = preprocess_data(X_train_intp, X_test_intp)
X_train_intp_xt, X_test_intp_xt, _, _ = preprocessed_data['xt']
X_train_intp_dx, X_test_intp_dx, _, _ = preprocessed_data['dx']
X_train_intp_xf, X_test_intp_xf, _, _ = preprocessed_data['xf']

preprocessed_data = preprocess_data(X_train_aug, X_test_aug)
X_train_aug_xt, X_test_aug_xt, _, _ = preprocessed_data['xt']
X_train_aug_dx, X_test_aug_dx, _, _ = preprocessed_data['dx']
X_train_aug_xf, X_test_aug_xf, _, _ = preprocessed_data['xf']

X_train = [X_train_intp_xt, X_train_intp_dx, X_train_intp_xf]
X_train_aug = [X_train_aug_xt, X_train_aug_dx, X_train_aug_xf]

##
pretrain_dataset = Load_Dataset(X_train, X_train_aug, y_train, 'pretrain')
prevalid_dataset = Load_Dataset(X_train, X_train_aug, y_train, 'prevalid')
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size_pretrain, shuffle=True, drop_last=False)
prevalid_loader = DataLoader(prevalid_dataset, batch_size=args.batch_size_pretrain, shuffle=False, drop_last=False)

##
os.makedirs(f'model_pretrain', exist_ok=True)
os.makedirs(f'model_pretrain/{args.data_name}', exist_ok=True)

os.makedirs(f'out_pretrain', exist_ok=True)
os.makedirs(f'out_pretrain/{args.data_name}', exist_ok=True)

# Dimension reduction with PCA
if args.num_feature > 64:
    args.num_feature = 64

if torch.cuda.device_count() > 1:
    encoder = Encoder(args)
    encoder = nn.DataParallel(encoder).to(device)
else:
    encoder = Encoder(args).to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=10, verbose=False)

loss_list = []
best_valid_loss = float('inf')
best_model_path = f'model_pretrain/{args.data_name}/{args.data_name}_{args.seed}.pth'
output_file = f'out_pretrain/{args.data_name}/{args.data_name}_{args.seed}'

# Early stopping parameters
patience = 20
early_stop_counter = 0

print(args)
for epoch in range(1, args.epochs_pretrain + 1):
    encoder.train()
    train_loss = train(args, encoder, None, encoder_optimizer, None, pretrain_loader, mode='pretrain', device=device)
    
    encoder.eval()
    with torch.no_grad():
        valid_loss = test(args, encoder, None, prevalid_loader, mode='pretrain', device=device)
    
    scheduler.step(valid_loss)
    
    print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}')
    loss_list.append([train_loss, valid_loss])
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        early_stop_counter = 0
        print(f'[Saving model at epoch {epoch} with validation loss {valid_loss:.4f}]')
        
        # Save the model state, optimizer state, and current epoch
        torch.save({
            'epoch': epoch,
            'args': args,
            # 'encoder_state_dict': encoder.state_dict(),
            'encoder_state_dict': encoder.module.state_dict() if isinstance(encoder, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else encoder.state_dict(),
            'optimizer_state_dict': encoder_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_list': loss_list,
            'best_valid_loss': best_valid_loss
        }, best_model_path)
    else:
        early_stop_counter += 1
    
    # Check for early stopping
    if early_stop_counter >= patience:
        print(f'Early stopping triggered at epoch {epoch}')
        break

# Save final results
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as f:
    pickle.dump([args, loss_list], f)

print(f"Training completed. Best validation loss: {best_valid_loss:.4f}")
