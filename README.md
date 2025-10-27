# Multi-Stain Attention Multiple Instance Learning for Prognosis Prediction in Esophageal Squamous Cell Carcinoma

## Installation

Create a conda environment and install the requirements

```shell
conda env create -f environment.yml
```

Activate the msamil environment

```shell
conda activate msamil
```

Install PyTorch

```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```


## WSI Segmentation and Patching

The following directory structure is required for data.

```bash
├── root
    ├── DATA_DIRECTORY
        ├── HE
        ├── CD4
        └── ...
    └── ANNOTATION_DIR(optional)
```
Before running the script, set the path in the script itself.

```shell
python create_patches_fp.py --source DATA_DIRECTORY/HE --save_dir COORDS_DIR --patch_size patch_size --step_size step_size --preset escc.csv --seg --patch --stitch
```

## Feature Extraction

```shell
python extract_features_m.py --model_name gigapath --coords_dir COORDS_DIR --data_slide_dir DATA_DIRECTORY --patients_csv patients.csv --feat_dir DATA_ROOT_DIR_gigapath --batch_size 1024 --slide_ext .tiff
```

This creates a following directory structure

```bash
├── root
    ├── DATA_DIRECTORY
        ├── HE
        ├── CD4
        └── ...
    ├── ANNOTATION_DIR
    ├── COORDS_DIR
    ├── DATA_ROOT_DIR
        ├── DATA_ROOT_DIR_gigapath
        ├── DATA_ROOT_DIR_uni
        └── ...
```

## Training Splits

For evaluating the algorithm's performance, multiple folds (e.g. 5-fold) of train/val splits can be used.

```shell
python create_splits_seq.py --task task_1_M --seed SEED --k 5 --method balanced
```

## Training

To train a model, use the following command.

```shell
python main.py --drop_out 0.25 --early_stopping --lr 5e-6 --k 5 --exp_code escc_gigapath_msamil_M9 --weighted_sample --bag_loss focal --task task_1_M --model_type msamil --log_data --data_root_dir DATA_ROOT_DIR_gigapath --embed_dim 1536
```

## Evaluation

To evaluate a model, use the following command.

```shell
python eval.py --k 5 --models_exp_code escc_gigapath_msamil_M9_s1 --save_exp_code escc_gigapath_msamil_M9_s1_cv --task task_1_M --model_type msamil --results_dir results --data_root_dir DATA_ROOT_DIR_gigapath --embed_dim 1024
```
