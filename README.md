# MSAMIL

Esophageal squamous cell carcinoma (ESCC) is a highly aggressive cancer with poor prognosis. Accurate prognostic models are essential for guiding personalized treatment strategies. However, most existing models rely solely on hematoxylin-eosin (HE) stained images, without fully leveraging the complementary information provided by multiple immunohistochemical (IHC) stains. IHC markers are crucial for capturing immune-related features within the tumor microenvironment, which are closely associated with patient outcomes. To address this limitation, we propose Multi-Stain Attention Multiple Instance Learning (MSAMIL), a novel framework that integrates information from multiple staining modalities—including key IHC markers such as CD4 and PD-L1 alongside HE—to better capture tumor-immune interactions and tissue heterogeneity. We evaluate MSAMIL on a private dataset, which includes whole slide images (WSIs) from nine staining modalities across 208 ESCC patients. The model predicts 4-year disease-free survival (DFS) with superior accuracy and F1 score compared to existing methods. We further observe that the choice of feature extractor significantly affects performance: domain-specific backbones such as Prov-Gigapath consistently outperform generic encoders, underscoring the importance of tailored pretraining for histopathological representation learning. Ablation confirms the fusion module is critical for modeling cross-modality interactions. Modality contribution analysis shows that CD4 and PD-L1 contribute most significantly, underscoring their importance in capturing tumor immune microenvironment (TIME) features relevant to prognosis. MSAMIL thus provides a powerful tool for ESCC prognostic prediction and may support personalized treatment decisions.



**Multi-Stain Attention Multiple Instance Learning for Prognosis Prediction in Esophageal Squamous Cell Carcinoma**

Xiaoya Fan, Zhengde Jia, Yuanyuan Han, Ruishan Geng, Xiaoyan Li, and Zhong Wang



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


## Datasets

The following directory structure is required for data.

```bash
├── root
    ├── DATA_DIRECTORY
        ├── HE                             # Hematoxylin-Eosin WSIs, e.g., slide_001.tiff, slide_002.tiff
        ├── CD4                            # CD4 IHC WSIs, e.g., slide_001.tiff, slide_002.tiff
        └── ...
    └── ANNOTATION_DIR(optional)           # Optional, e.g., slide_001.json, slide_002.json
```

## WSI Segmentation and Patching

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
