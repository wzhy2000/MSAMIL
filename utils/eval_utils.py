import numpy as np
import torch
from models.model import *
import os
import pandas as pd
from utils.utils import *
from sklearn.metrics import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'msdlm':
        model = MSDLM(dropout=args.drop_out, n_classes=args.n_classes, n_modalities=args.n_modalities,
                      embed_dim=args.embed_dim, method=args.method)
    elif args.model_type == 'msamil':
        model = MSAMIL(mil_method=args.mil_method, dropout = args.drop_out, n_classes=args.n_classes, n_modalities=args.n_modalities,
                       embed_dim=args.embed_dim)
    elif args.model_type == 'abmil':
        model = DeepMIL(embed_dim=args.embed_dim, n_classes=args.n_classes, dropout=args.drop_out)
    elif args.model_type == 'dsmil':
            model = DSMIL(embed_dim=args.embed_dim, n_classes=args.n_classes, dropout=args.drop_out)
    elif args.model_type == 'transmil':
        model = TransMIL(embed_dim=args.embed_dim, n_classes=args.n_classes)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(dropout = args.drop_out, n_classes = args.n_classes, embed_dim=args.embed_dim)
        else:
            model = MIL_fc(dropout = args.drop_out, n_classes = args.n_classes, embed_dim=args.embed_dim)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path, k_idx):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    results_dict = summary(model, loader, args, k_idx)
    if args.shap:
        from utils.shap_utils import shap_analysis_msamil
        shap_matrix, X_shap = shap_analysis_msamil(model, loader, save_dir=os.path.join(args.save_dir, 'shap_plots'), fold_idx=k_idx)
        return model, results_dict, shap_matrix, X_shap
    return model, results_dict, None, None

def summary(model, loader, args, k_idx):
    if not os.path.isdir(os.path.join(args.save_dir, 'confusion_matrix/')):
        os.mkdir(os.path.join(args.save_dir, 'confusion_matrix/'))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    val_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    patient_ids = loader.dataset.patient_data['Patient_ID']
    patient_results = {}
    tiles_predict_label = []
    tiles_true_label = []
    
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        patient_id = patient_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        if Y_hat.shape == torch.Size([1]):
            tiles_predict_label.extend(Y_hat.cpu())
        else:
            tiles_predict_label.extend(Y_hat.cpu().squeeze(1))
        # tiles_predict_label.extend(Y_hat.cpu())
        tiles_true_label.extend(label.cpu())
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({patient_id: {'Patient_ID': np.array(patient_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        val_error += error

    del data
    val_error /= len(loader)

    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
            fpr,tpr,_ = roc_curve(all_labels,all_probs[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'Patient_ID': patient_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})

    conf_matrix = confusion_matrix(tiles_true_label, tiles_predict_label)
    tn, fp, fn, tp = conf_matrix.ravel()
    acc = accuracy_score(tiles_true_label, tiles_predict_label)
    specificity = tn / (tn + fp)
    precision = precision_score(tiles_true_label, tiles_predict_label)
    recall = recall_score(tiles_true_label, tiles_predict_label)
    macroF1 = f1_score(tiles_true_label, tiles_predict_label,  average='macro')
    F1 = f1_score(tiles_true_label, tiles_predict_label)

    df = pd.DataFrame(results_dict)
    inference_results = {'patient_results': patient_results, 'val_error': val_error,
                     'auc': auc_score, 'aucs': aucs,'df':df, 'acc': acc, 'specificity':specificity, 'precision': precision,
                         'recall': recall, 'macroF1': macroF1,'F1': F1}
    return inference_results



def plot_auprc(precision, recall, auc, path=None, title='AuPrc', save_fig=False):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(precision, recall, color='darkorange', lw=0.7, label=f'PRC curve (area = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.7, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Precision Recall Curve example')
    plt.legend(loc='lower right')
    if save_fig:
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, title + '.png'))
    else:
        plt.show()
    plt.close()


def plot_auroc(fpr, tpr, auc, path=None, title='AuRoc', save_fig=False):
    fig = plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, color='darkorange', lw=0.7, label=f'ROC curve (area = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.7, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    if save_fig:
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, title + '.png'))
    else:
        plt.show()
    plt.close()