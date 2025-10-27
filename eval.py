from __future__ import print_function

import argparse
import torch
import os
import pandas as pd
from utils.utils import *
from utils.plot import *
from dataset_modules.dataset_generic import Generic_features_Dataset
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, default='msamil', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'all'], default='val')
parser.add_argument('--task', type=str, choices=['task_1_M', 'task_2_HE', 'task_3_CD4', 'task_4_CD8',
                                                'task_4_CD8', 'task_5_CD20', 'task_6_CD56', 'task_7_CD68',
                                                'task_8_CD163', 'task_9_FOXP3', 'task_10_PDL1', 'task_11_M4', 'task_12_M2'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--no_plot', action='store_true', default=False)
parser.add_argument('--method', type=str, default='mean')
parser.add_argument('--mil_method', type=str, default='abmil')
parser.add_argument('--shap', action='store_true', help='whether to compute SHAP values')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_M':
    args.n_classes=2
    args.n_modalities=9
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ["CD4", "CD8", "CD20", "CD56", "CD68", "CD163", "FOXP3", "PDL1", "HE"],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])


elif args.task == 'task_2_HE':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['HE'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_3_CD4':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['CD4'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_4_CD8':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['CD8'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_5_CD20':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['CD20'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_6_CD56':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['CD56'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_7_CD68':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['CD68'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_8_CD163':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['CD163'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_9_FOXP3':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['FOXP3'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_10_PDL1':
    args.n_classes=2
    args.n_modalities=1
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ['PDL1'],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_11_M4':
    args.n_classes=2
    args.n_modalities=2
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ["CD4", "PDL1", "FOXP3", "CD163"],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

elif args.task == 'task_12_M2':
    args.n_classes=2
    args.n_modalities=2
    dataset = Generic_features_Dataset(csv_path = 'patients.csv',
                            data_dir= args.data_root_dir,
                            wsi_types = ["CD4", "PDL1"],
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'0':0, '1':1},
                            ignore=[])
    

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold + 1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'all': -1}

if __name__ == "__main__":
    all_auc = []
    all_acc = []
    all_specificity = []
    all_precision = []
    all_recall = []
    all_macroF1 = []
    all_F1 = []

    all_shap_matrices = []
    all_X_shap = []

    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, results_dict, shap_matrix, X_shap = eval(split_dataset, args, ckpt_paths[ckpt_idx], str(folds[ckpt_idx]))
        if shap_matrix is not None:
            all_shap_matrices.append(shap_matrix)
            all_X_shap.append(X_shap)


        for cls_idx in range(len(results_dict['aucs'])):
            print('class {} auc: {}'.format(cls_idx, results_dict['aucs'][cls_idx]))

        all_auc.append(results_dict['auc'])
        all_acc.append(results_dict['acc'])
        all_specificity.append(results_dict['specificity'])
        all_precision.append(results_dict['precision'])
        all_recall.append(results_dict['recall'])
        all_macroF1.append(results_dict['macroF1'])
        all_F1.append(results_dict['F1'])
        df = results_dict['df']
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    df_dict = {'folds': folds, 'val_auc': all_auc, 'val_acc': all_acc,
               'specificity':all_specificity, 'precision': all_precision,
                         'recall': all_recall, 'macroF1': all_macroF1, 'F1': all_F1}

    save_path = os.path.join(args.save_dir, 'summary.csv')
    final_df = pd.DataFrame(df_dict)
    if os.path.exists(save_path):
        final_df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        final_df.to_csv(save_path, index=False)
    
    if args.no_plot is False:
        df = pd.read_csv(save_path)
        mean_values = df.iloc[:, 1:].mean()
        mean_row = pd.DataFrame([['mean'] + mean_values.tolist()], columns=df.columns)
        mean_row.to_csv(save_path, mode='a', header=False, index=False)

        std_values = df.iloc[:, 1:].std()
        std_row = pd.DataFrame([['std'] + std_values.tolist()], columns=df.columns)
        std_row.to_csv(save_path, mode='a', header=False, index=False)

        plot_k_roc(args.save_dir, args.k)
        plot_k_prc(args.save_dir, args.k)
        plot_k_confusion_matrix(args.save_dir, args.k)


    if args.shap and len(all_shap_matrices) > 0:
        from utils.shap_utils import shap_summary_plot
        # 最后拼接所有 SHAP
        shap_matrix_all = np.concatenate(all_shap_matrices, axis=0)  # shape: [N, 9]
        X_shap_all = np.concatenate(all_X_shap, axis=0)              # shape: [N, 9]

        print(f"shap_matrix_all shape: {shap_matrix_all.shape}, X_shap_all shape: {X_shap_all.shape}")

        shap_data = {
            "shap_values": shap_matrix_all,        # shape: (n_samples, n_modalities)
            "features": X_shap_all        # shape: (n_samples, n_modalities)
        }

        save_path = os.path.join(args.save_dir, f"shap.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(shap_data, f)

        shap_summary_plot(shap_matrix_all, X_shap_all, save_dir=args.save_dir)
        