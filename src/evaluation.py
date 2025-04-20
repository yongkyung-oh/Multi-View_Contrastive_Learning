import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def repeat_if_batch_size_one(tensor):
    return torch.cat([tensor, tensor], dim=0) if tensor.size(0) == 1 else tensor


def get_features(args, encoder, loader, device):
    encoder.eval()
    feature_t, feature_d, feature_f = [], [], []
    y_all = []
    
    with torch.no_grad():
        for Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y in loader:
            tensors = [Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y]
            tensors = [repeat_if_batch_size_one(t.to(device)) for t in tensors]
            Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y = tensors
            
            ht, hd, hf, zt, zd, zf = encoder(Xt, dX, Xf)
            
            if args.feature == 'latent':
                feature_t.append(zt)
                feature_d.append(zd)
                feature_f.append(zf)
            elif args.feature == 'hidden':
                feature_t.append(ht)
                feature_d.append(hd)
                feature_f.append(hf)
            else:
                raise ValueError(f"Invalid feature type: {args.feature}")
            
            y_all.append(y)

    # Concatenate features and labels
    feature_t = torch.cat(feature_t, dim=0)
    feature_d = torch.cat(feature_d, dim=0)
    feature_f = torch.cat(feature_f, dim=0)
    y_all = torch.cat(y_all, dim=0)

    return feature_t, feature_d, feature_f, y_all


def get_clf_acc(args, encoder, clf, loader, device):
    encoder.eval()
    clf.eval()
    with torch.no_grad():
        logit_all, pred_all, y_all = [], [], []
        for Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y in loader:
            tensors = [Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y]
            tensors = [repeat_if_batch_size_one(t.to(device)) for t in tensors]
            Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y = tensors
            
            ht, hd, hf, zt, zd, zf = encoder(Xt, dX, Xf)
            if args.feature == 'latent':
                logit = clf(zt, zd, zf)
            elif args.feature == 'hidden':
                logit = clf(ht, hd, hf)
            pred = logit.detach().argmax(dim=1)
            logit_all.append(logit)
            pred_all.append(pred)
            y_all.append(y)
        logit_all = torch.cat(logit_all, dim=0)
        pred_all = torch.cat(pred_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        
        correct_predictions = torch.eq(pred_all.detach().cpu(), y_all.detach().cpu())
        return correct_predictions.sum().item() / len(y_all)


def get_clf_metrics(args, encoder, clf, loader, device):
    encoder.eval()
    clf.eval()
    with torch.no_grad():
        logit_all, pred_all, y_all = [], [], []
        for Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y in loader:
            tensors = [Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y]
            tensors = [repeat_if_batch_size_one(t.to(device)) for t in tensors]
            Xt, dX, Xf, Xt_aug, dX_aug, Xf_aug, y = tensors
            ht, hd, hf, zt, zd, zf = encoder(Xt, dX, Xf)
            if args.feature == 'latent':
                logit = clf(zt, zd, zf)
            elif args.feature == 'hidden':
                logit = clf(ht, hd, hf)
            pred = logit.detach().argmax(dim=1)
            logit_all.append(logit)
            pred_all.append(pred)
            y_all.append(y)
        
        logit_all = torch.cat(logit_all, dim=0)
        pred_all = torch.cat(pred_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        
        # Convert to numpy for sklearn metrics
        y_true = y_all.cpu().numpy()
        y_pred = pred_all.cpu().numpy()
        y_score = torch.softmax(logit_all, dim=1).cpu().numpy()
        
        # Calculate accuracy
        accuracy = (y_true == y_pred).mean()
        
        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Initialize AUROC and AUPRC
        auroc = None
        auprc = None
        
        # Check if more than one class is present
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1:
            if args.num_target == 2:
                # Binary classification
                auroc = roc_auc_score(y_true, y_score[:, 1])
                auprc = average_precision_score(y_true, y_score[:, 1])
            else:
                # Multi-class classification
                y_true_bin = label_binarize(y_true, classes=range(args.num_target))
                try:
                    auroc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
                    auprc = average_precision_score(y_true_bin, y_score, average='macro')
                except:
                    pass
        else:
            print(f"Warning: Only one class present in the evaluation set (Class {unique_classes[0]}). AUROC and AUPRC are undefined.")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'auroc': auroc,
            'auprc': auprc
        }