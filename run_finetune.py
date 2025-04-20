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


#
if str(args.loss_type) not in ['ALL', 'TDF'] and float(args.lam) == 0.0:
    print("Error: check loss_type and lam")
    sys.exit(1)

#
args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

#
if str(args.loss_type) not in ['ALL', 'TDF'] and args.horizon_len not in [0, 16]:
    print("Error: check loss_type and horizon")
    sys.exit(1)

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
preprocessed_data = preprocess_data(X_train_intp, X_val_intp)
X_train_intp_xt, X_val_intp_xt, _, _ = preprocessed_data['xt']
X_train_intp_dx, X_val_intp_dx, _, _ = preprocessed_data['dx']
X_train_intp_xf, X_val_intp_xf, _, _ = preprocessed_data['xf']

preprocessed_data = preprocess_data(X_train_intp, X_test_intp)
X_train_intp_xt, X_test_intp_xt, _, _ = preprocessed_data['xt']
X_train_intp_dx, X_test_intp_dx, _, _ = preprocessed_data['dx']
X_train_intp_xf, X_test_intp_xf, _, _ = preprocessed_data['xf']

X_train = [X_train_intp_xt, X_train_intp_dx, X_train_intp_xf]
X_valid = [X_val_intp_xt, X_val_intp_dx, X_val_intp_xf]
X_test = [X_test_intp_xt, X_test_intp_dx, X_test_intp_xf]

train_dataset = Load_Dataset(X_train, X_train, y_train, 'finetune')
valid_dataset = Load_Dataset(X_valid, X_valid, y_val, 'test')
test_dataset = Load_Dataset(X_test, X_test, y_test, 'test')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size_finetune, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_finetune, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size_finetune, shuffle=False, drop_last=False)

##
os.makedirs(f'out_finetune', exist_ok=True)
os.makedirs(f'out_finetune/{args.data_name}', exist_ok=True)

# Dimension reduction with PCA
if args.num_feature > 64:
    args.num_feature = 64

##
if len(args.loss_type) == 3:
    K = 1 # 5
else:
    K = 1

##
monitoring_metric = 'accuracy'

print(args)
for k in range(K):
    ## Run -- finetune
    best_model_path = f'model_pretrain/{args.data_name}/{args.data_name}_{args.seed}.pth'
    if torch.cuda.device_count() > 1:
        encoder = Encoder(args)
        encoder = load_encoder(encoder, best_model_path, args.num_feature)
        encoder = nn.DataParallel(encoder).to(device)
        clf = Classifier(args)
        clf = nn.DataParallel(clf).to(device)
    else:
        encoder = Encoder(args).to(device)
        clf = Classifier(args).to(device)
    
    for param in encoder.parameters():
        param.requires_grad = True
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    
    loss_list = []
    metric_list = []
    best_valid_mm = 0
    best_valid_loss = float('inf')
    output_file = f'out_finetune/{args.data_name}/{args.data_name}_{args.seed}_{args.feature}_{args.loss_type}_{args.lam}_{k}_finetune'
    
    # Early stopping parameters
    patience = 20
    early_stop_counter = 0
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping this run.")
    else:
        # Run experiment
        for epoch in range(1, args.epochs_finetune + 1):
            train_loss, train_loss_c = train(args, encoder, clf, encoder_optimizer, clf_optimizer, train_loader, mode='finetune', device=device)
            valid_loss, valid_loss_c = test(args, encoder, clf, valid_loader, mode='valid', device=device)
            test_loss, test_loss_c = test(args, encoder, clf, test_loader, mode='test', device=device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, Test Loss = {test_loss:.4f}')
            print(f'Epoch {epoch}: Train Loss_c = {train_loss_c:.4f}, Validation Loss_c = {valid_loss_c:.4f}, Test Loss_c = {test_loss_c:.4f}')
            loss_list.append([train_loss, valid_loss, test_loss, train_loss_c, valid_loss_c, test_loss_c])
            
            # Compute metrics
            train_metric = get_clf_metrics(args, encoder, clf, train_loader, device)
            valid_metric = get_clf_metrics(args, encoder, clf, valid_loader, device)
            test_metric = get_clf_metrics(args, encoder, clf, test_loader, device)
            
            train_mm = train_metric[monitoring_metric]
            valid_mm = valid_metric[monitoring_metric]
            test_mm = test_metric[monitoring_metric]
            
            print(f'Epoch {epoch}: Train Metric = {train_mm:.4f}, Validation Metric = {valid_mm:.4f}, Test Metric = {test_mm:.4f}')
            metric_list.append([train_metric, valid_metric, test_metric])
            
            # Learning rate scheduling
            encoder_scheduler.step(valid_mm)
            clf_scheduler.step(valid_mm)
            
            # if valid_loss_c < best_valid_loss:
            #     best_valid_loss = valid_loss_c
            if valid_mm > best_valid_mm:
                best_valid_mm = valid_mm
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump([args, loss_list, metric_list], f)
        

    ## Run -- freeze
    best_model_path = f'model_pretrain/{args.data_name}/{args.data_name}_{args.seed}.pth'
    if torch.cuda.device_count() > 1:
        encoder = Encoder(args)
        encoder = load_encoder(encoder, best_model_path, args.num_feature)
        encoder = nn.DataParallel(encoder).to(device)
        clf = Classifier(args)
        clf = nn.DataParallel(clf).to(device)
    else:
        encoder = Encoder(args).to(device)
        clf = Classifier(args).to(device)
    
    for name, param in encoder.named_parameters():
        if 'input_layer' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    
    loss_list = []
    metric_list = []
    best_valid_mm = 0
    best_valid_loss = float('inf')
    output_file = f'out_finetune/{args.data_name}/{args.data_name}_{args.seed}_{args.feature}_{args.loss_type}_{args.lam}_{k}_freeze'
    
    # Early stopping parameters
    patience = 20
    early_stop_counter = 0
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping this run.")
    else:
        # Run experiment
        for epoch in range(1, args.epochs_finetune + 1):
            train_loss, train_loss_c = train(args, encoder, clf, encoder_optimizer, clf_optimizer, train_loader, mode='freeze', device=device)
            valid_loss, valid_loss_c = test(args, encoder, clf, valid_loader, mode='valid', device=device)
            test_loss, test_loss_c = test(args, encoder, clf, test_loader, mode='test', device=device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, Test Loss = {test_loss:.4f}')
            print(f'Epoch {epoch}: Train Loss_c = {train_loss_c:.4f}, Validation Loss_c = {valid_loss_c:.4f}, Test Loss_c = {test_loss_c:.4f}')
            loss_list.append([train_loss, valid_loss, test_loss, train_loss_c, valid_loss_c, test_loss_c])
            
            # Compute metrics
            train_metric = get_clf_metrics(args, encoder, clf, train_loader, device)
            valid_metric = get_clf_metrics(args, encoder, clf, valid_loader, device)
            test_metric = get_clf_metrics(args, encoder, clf, test_loader, device)
            
            train_mm = train_metric[monitoring_metric]
            valid_mm = valid_metric[monitoring_metric]
            test_mm = test_metric[monitoring_metric]
            
            print(f'Epoch {epoch}: Train Metric = {train_mm:.4f}, Validation Metric = {valid_mm:.4f}, Test Metric = {test_mm:.4f}')
            metric_list.append([train_metric, valid_metric, test_metric])
            
            # Learning rate scheduling
            encoder_scheduler.step(valid_mm)
            clf_scheduler.step(valid_mm)
            
            # if valid_loss_c < best_valid_loss:
            #     best_valid_loss = valid_loss_c
            if valid_mm > best_valid_mm:
                best_valid_mm = valid_mm
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump([args, loss_list, metric_list], f)
        

    ## Run -- finetune -- baseline
    if torch.cuda.device_count() > 1:
        encoder = Encoder(args)
        encoder = nn.DataParallel(encoder).to(device)
        clf = Classifier(args)
        clf = nn.DataParallel(clf).to(device)
    else:
        encoder = Encoder(args).to(device)
        clf = Classifier(args).to(device)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    
    loss_list = []
    metric_list = []
    best_valid_mm = 0
    best_valid_loss = float('inf')
    # best_model_path = f'model_finetune/{args.data_name}/{args.data_name}_{args.seed}_{args.feature}_{args.loss_type}_{args.lam}_{k}_baseline.pth'
    output_file = f'out_finetune/{args.data_name}/{args.data_name}_{args.seed}_{args.feature}_{args.loss_type}_{args.lam}_{k}_baseline'
    
    # Early stopping parameters
    patience = 20
    early_stop_counter = 0
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping this run.")
    else:
        # Run experiment
        for epoch in range(1, args.epochs_finetune + 1):
            train_loss, train_loss_c = train(args, encoder, clf, encoder_optimizer, clf_optimizer, train_loader, mode='baseline', device=device)
            valid_loss, valid_loss_c = test(args, encoder, clf, valid_loader, mode='valid', device=device)
            test_loss, test_loss_c = test(args, encoder, clf, test_loader, mode='test', device=device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, Test Loss = {test_loss:.4f}')
            print(f'Epoch {epoch}: Train Loss_c = {train_loss_c:.4f}, Validation Loss_c = {valid_loss_c:.4f}, Test Loss_c = {test_loss_c:.4f}')
            loss_list.append([train_loss, valid_loss, test_loss, train_loss_c, valid_loss_c, test_loss_c])
            
            # Compute metrics
            train_metric = get_clf_metrics(args, encoder, clf, train_loader, device)
            valid_metric = get_clf_metrics(args, encoder, clf, valid_loader, device)
            test_metric = get_clf_metrics(args, encoder, clf, test_loader, device)
            
            train_mm = train_metric[monitoring_metric]
            valid_mm = valid_metric[monitoring_metric]
            test_mm = test_metric[monitoring_metric]
            
            print(f'Epoch {epoch}: Train Metric = {train_mm:.4f}, Validation Metric = {valid_mm:.4f}, Test Metric = {test_mm:.4f}')
            metric_list.append([train_metric, valid_metric, test_metric])
            
            # Learning rate scheduling
            encoder_scheduler.step(valid_mm)
            clf_scheduler.step(valid_mm)
            
            # if valid_loss_c < best_valid_loss:
            #     best_valid_loss = valid_loss_c
            if valid_mm > best_valid_mm:
                best_valid_mm = valid_mm
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump([args, loss_list, metric_list], f)
