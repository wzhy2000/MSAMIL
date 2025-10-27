import os
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_M', 'task_2_HE', 'task_3_CD4', 'task_4_CD8',
                                                'task_4_CD8', 'task_5_CD20', 'task_6_CD56', 'task_7_CD68',
                                                'task_8_CD163', 'task_9_FOXP3', 'task_10_PDL1', 'task_11_M4', 'task_12_M2'])
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='fraction of labels for validation (default: 0.2)')
parser.add_argument('--method', type=str, default='montecarlo', choices=['montecarlo', 'balanced'])

args = parser.parse_args()


if args.task == 'task_1_M':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])

    
elif args.task == 'task_2_HE':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_3_CD4':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_4_CD8':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_5_CD20':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_6_CD56':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_7_CD68':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_8_CD163':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_9_FOXP3':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_10_PDL1':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_11_M4':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    


elif args.task == 'task_12_M2':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'patients.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'no_relapse':0, 'relapse':1},
                            ignore=[])
    

elif args.task == 'task_22_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_voting='maj',
                            ignore=[])
    

else:
    raise NotImplementedError

num_patients_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_patients_cls * args.val_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, label_frac=lf, method=args.method)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



