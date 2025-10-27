#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


# Create patches
python create_patches_fp.py --source DATA_DIRECTORY/HE --save_dir COORDS_DIR --patch_size patch_size --step_size step_size --preset escc.csv --seg --patch --stitch

# Feature extraction
python extract_features.py --model_name resnet18 --coords_dir COORDS_DIR --data_slide_dir DATA_DIRECTORY --patients_csv patients.csv --feat_dir DATA_ROOT_DIR_resnet18 --batch_size 1024 --slide_ext .tiff
python extract_features.py --model_name resnet50 --coords_dir COORDS_DIR --data_slide_dir DATA_DIRECTORY --patients_csv patients.csv --feat_dir DATA_ROOT_DIR_resnet50 --batch_size 1024 --slide_ext .tiff
python extract_features.py --model_name uni_v2 --coords_dir COORDS_DIR --data_slide_dir DATA_DIRECTORY --patients_csv patients.csv --feat_dir DATA_ROOT_DIR_uni_v2 --batch_size 1024 --slide_ext .tiff
python extract_features.py --model_name virchow2 --coords_dir COORDS_DIR --data_slide_dir DATA_DIRECTORY --patients_csv patients.csv --feat_dir DATA_ROOT_DIR_virchow2 --batch_size 1024 --slide_ext .tiff
python extract_features.py --model_name gigapath --coords_dir COORDS_DIR --data_slide_dir DATA_DIRECTORY --patients_csv patients.csv --feat_dir DATA_ROOT_DIR_gigapath --batch_size 1024 --slide_ext .tiff

# Training Splits
python create_splits_seq.py --task task_1_M --seed SEED --k 5 --method balanced

# Training and evaluation
python main.py --drop_out 0.25 --early_stopping --lr 5e-6 --k 5 --exp_code escc_gigapath_msamil_M9 --weighted_sample --bag_loss focal --task task_1_M --model_type msamil --log_data --data_root_dir DATA_ROOT_DIR_gigapath --embed_dim 1536
python eval.py --k 5 --models_exp_code escc_gigapath_msamil_M9_s1 --save_exp_code escc_gigapath_msamil_M9_s1_cv --task task_1_M --model_type msamil --results_dir results --data_root_dir DATA_ROOT_DIR_gigapath --embed_dim 1536
