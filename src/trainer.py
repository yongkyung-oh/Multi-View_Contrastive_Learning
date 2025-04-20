import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pytorch_metric_learning import losses
from tqdm import tqdm


def add_weight_regularization(model, l1_scale=0.0, l2_scale=0.01):
    l1_reg, l2_reg = 0.0, 0.0

    for parameter in model.parameters():
        if parameter.requires_grad:
            l1_reg += l1_scale * parameter.abs().sum()
            l2_reg += l2_scale * parameter.pow(2).sum()
            
    return l1_reg + l2_reg


def get_loss_by_type(loss_type, loss_t, loss_d, loss_f):
    loss_dict = {
        'ALL': loss_t + loss_d + loss_f,
        'TDF': loss_t + loss_d + loss_f,
        'TD': loss_t + loss_d,
        'TF': loss_t + loss_f,
        'DF': loss_d + loss_f,
        'T': loss_t,
        'D': loss_d,
        'F': loss_f
    }
    if loss_type not in loss_dict:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss_dict[loss_type]


def train(args, encoder, clf, encoder_optimizer, clf_optimizer, loader, mode='pretrain', device='cuda'):
    encoder.train() if mode != 'freeze' else encoder.eval() 
    clf.train() if mode != 'pretrain' else None

    if mode == 'pretrain':
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True
        # clf.eval()
    elif mode == 'finetune':
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True
        clf.train()
    elif mode == 'freeze':
        encoder.eval()
        for name, param in encoder.named_parameters():
            if 'input_layer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        clf.train()
    
    scaler = GradScaler()
    
    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_loss_c = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Training ({mode})")
    for batch in pbar:      
        xt, dx, xf, xt_aug, dx_aug, xf_aug, y = [t.float().to(device) for t in batch]
        
        encoder_optimizer.zero_grad()
        if mode != 'pretrain':
            clf_optimizer.zero_grad()
        
        with autocast(enabled=True):
            ht, hd, hf, zt, zd, zf = encoder(xt, dx, xf)
            ht_aug, hd_aug, hf_aug, zt_aug, zd_aug, zf_aug = encoder(xt_aug, dx_aug, xf_aug)
            
            loss_t = info_criterion(zt, zt_aug)
            loss_d = info_criterion(zd, zd_aug)
            loss_f = info_criterion(zf, zf_aug)
            
            loss = get_loss_by_type(args.loss_type, loss_t, loss_d, loss_f) + add_weight_regularization(encoder)

            for name, param in encoder.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"encoder: NaN or Inf in gradients of {name}")
            
            if mode != 'pretrain':
                logit = clf(zt, zd, zf) if args.feature == 'latent' else clf(ht, hd, hf)
                loss_c = criterion(logit, y.long())
                loss = args.lam * loss + loss_c + add_weight_regularization(clf)

                for name, param in clf.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"clf: NaN or Inf in gradients of {name}")

        scaler.scale(loss).backward()
        
        # # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=10.0)
        # if mode != 'pretrain':
        #     torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=10.0)

        if mode != 'freeze':
            scaler.step(encoder_optimizer)
        if mode != 'pretrain':
            scaler.step(clf_optimizer)
        
        scaler.update()
        
        total_loss += loss.item() * xt.size(0)
        if mode != 'pretrain':
            total_loss_c += loss_c.item() * xt.size(0)
        total_samples += xt.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'loss_t': loss_t.item(), 'loss_d': loss_d.item(), 'loss_f': loss_f.item()})
    
    avg_loss = total_loss / total_samples
    avg_loss_c = total_loss_c / total_samples
    
    if mode == 'pretrain':
        return avg_loss
    else:
        return avg_loss, avg_loss_c        


def test(args, encoder, clf, loader, mode='pretrain', device='cuda'):
    encoder.eval()
    clf.eval() if mode != 'pretrain' else None

    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_loss_c = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Testing ({mode})")
        for batch in pbar:      
            xt, dx, xf, xt_aug, dx_aug, xf_aug, y = [t.float().to(device) for t in batch]

            with autocast(enabled=True):
                ht, hd, hf, zt, zd, zf = encoder(xt, dx, xf)
                ht_aug, hd_aug, hf_aug, zt_aug, zd_aug, zf_aug = encoder(xt_aug, dx_aug, xf_aug)
                
                loss_t = info_criterion(zt, zt_aug)
                loss_d = info_criterion(zd, zd_aug)
                loss_f = info_criterion(zf, zf_aug)
                
                loss = get_loss_by_type(args.loss_type, loss_t, loss_d, loss_f) + add_weight_regularization(encoder)
                
                if mode != 'pretrain':
                    logit = clf(zt, zd, zf) if args.feature == 'latent' else clf(ht, hd, hf)
                    loss_c = criterion(logit, y.long())
                    loss = args.lam * loss + loss_c + add_weight_regularization(clf)
            
            total_loss += loss.item() * xt.size(0)
            if mode != 'pretrain':
                total_loss_c += loss_c.item() * xt.size(0)
            total_samples += xt.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'loss_t': loss_t.item(), 'loss_d': loss_d.item(), 'loss_f': loss_f.item()})
    
    avg_loss = total_loss / total_samples
    avg_loss_c = total_loss_c / total_samples
    
    if mode == 'pretrain':
        return avg_loss
    else:
        return avg_loss, avg_loss_c 


## Pretrained model loader
def remove_module_prefix(state_dict):
    # Prevent multi-gpu -> single-gpu errors
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the first 7 characters ('module.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_encoder(encoder, checkpoint_path, new_num_feature=None):
    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # If it's a full checkpoint (not just the state_dict), extract the encoder_state_dict
    if isinstance(state_dict, dict) and 'encoder_state_dict' in state_dict:
        state_dict = state_dict['encoder_state_dict']
    
    # Remove the 'module.' prefix from multi-GPU training, if present
    state_dict = remove_module_prefix(state_dict)

    # If new_num_feature is specified and different from the loaded weights
    if new_num_feature is not None:
        input_layers = ['input_layer_t', 'input_layer_d', 'input_layer_f']
        for layer_name in input_layers:
            # Get old weights and biases
            old_weight = state_dict[f'{layer_name}.weight']
            old_bias = state_dict[f'{layer_name}.bias']
            old_num_feature = old_weight.size(1)
            
            if new_num_feature != old_num_feature:
                # Initialize new weights with correct dimensions
                new_weight = nn.Linear(new_num_feature, old_weight.size(0)).weight.data
                nn.init.xavier_uniform_(new_weight)
                
                # Replace weights and biases in the state_dict
                state_dict[f'{layer_name}.weight'] = new_weight
                state_dict[f'{layer_name}.bias'] = torch.zeros_like(old_bias)
    
    # Correct any weights that are NaN or Inf
    for key, param in state_dict.items():
        if torch.is_tensor(param):
            if torch.isnan(param).any() or torch.isinf(param).any():
                # Replace NaN and Inf with zero
                param = torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0)
                state_dict[key] = param
                
    # Remove
    keys_to_remove = [key for key in state_dict.keys() if 'q_func' in key]
    for key in keys_to_remove:
        state_dict.pop(key)
    
    # Load the modified state_dict into the encoder
    encoder.load_state_dict(state_dict, strict=False)
    
    return encoder

