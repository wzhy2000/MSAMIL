import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn as nn
import seaborn as sns
import pickle


class MSAMILBagWrapper(nn.Module):
    def __init__(self, model, num_modalities=9, modality_dim=1536):
        super(MSAMILBagWrapper, self).__init__() 
        self.model = model
        self.num_modalities = num_modalities
        self.modality_dim = modality_dim

    def forward(self, h_tensor):
        modality_list = []
        for i in range(self.num_modalities):
            start = i * self.modality_dim
            end = (i + 1) * self.modality_dim
            modality_list.append(h_tensor[:, start:end])  # 每个模态: [50, 1536]
        logits, probs, Y_hat, A_raw, _ = self.model(modality_list)
        # print(f"A_raw shape: {A_raw.shape}")  # A_raw shape: torch.Size([2, 50])
        return logits  # shape: [1, n_class]
    

def get_attention_from_model(model, h_list, device):
    with torch.no_grad():
        model.eval()
        h_list = [x.to(device) for x in h_list]
        _, _, _, A_raw, _ = model(h_list)  # 原模型必须返回 A_raw
    return A_raw[1].detach().cpu().numpy()  # shape = [n_patches]


def compute_attention_weighted_shap(shap_values_class, attention_weights):
    """
    shap_values_class: [n_patches, n_features]
    attention_weights: [n_patches]
    return: [n_features] — WSI-level SHAP 值
    """
    attention_weights = attention_weights / np.sum(attention_weights)
    return np.sum(shap_values_class * attention_weights[:, np.newaxis], axis=0)


def plot_modality_bar(modality_shap, modality_names, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.bar(modality_names, modality_shap)
    plt.ylabel("Mean |SHAP value|")
    plt.title("WSI-level Modality SHAP Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"modality_bar saved to {save_path}")


# def compute_attention_weighted_modality_shap(shap_values_class, attention_weights, modality_dim=1536, num_modalities=9):
#     """
#     返回：WSI 级别，每种染色模态的重要性（绝对值平均）
#     输出: np.array, shape = [9]
#     """
#     attention_weights = attention_weights / np.sum(attention_weights)
#     shap_wsi = np.sum(shap_values_class * attention_weights[:, np.newaxis], axis=0)  # shape: [13824]
    
#     modality_scores = []
#     for i in range(num_modalities):
#         start = i * modality_dim
#         end = (i + 1) * modality_dim
#         vals = shap_wsi[start:end]
#         score = np.mean(np.abs(vals))  # 绝对值平均
#         modality_scores.append(score)

#     return np.array(modality_scores)  # [9]


def plot_shap_cluster_heatmap(shap_values, save_path, modality_dim=1536, num_modalities=9):
    """
    shap_values: np.array of shape (n_samples, 13824), 单类别SHAP值（如 class 1）
    """
    n_samples = shap_values.shape[0]
    modality_shap = []

    for i in range(n_samples):
        patch_vals = shap_values[i]  # shape: [13824]
        modality_means = []
        for m in range(num_modalities):
            start = m * modality_dim
            end = (m + 1) * modality_dim
            val = np.mean(np.abs(patch_vals[start:end]))
            modality_means.append(val)
        modality_shap.append(modality_means)

    modality_shap = np.array(modality_shap)  # [n_samples, 9]

    plt.figure(figsize=(10, 8))
    sns.clustermap(modality_shap, cmap="viridis", yticklabels=False, 
                   xticklabels=["CD4", "CD8", "CD20", "CD56", "CD68", "CD163", "FOXP3", "PDL1", "HE"])
    plt.savefig(save_path)
    plt.close()

    print(f"cluster_heatmap saved to {save_path}")


def shap_analysis_msamil(model, loader, save_dir, fold_idx):
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    modality_names = ["CD4", "CD8", "CD20", "CD56", "CD68", "CD163", "FOXP3", "PDL1", "HE"]

    all_shap_wsi = []
    input_feat_list = []  # 用于存储每个病人的输入特征

    # step 1: 提取特定病人数据
    for idx, ((data, label), pid) in enumerate(zip(loader, loader.dataset.patient_data['Patient_ID'])):
        print(f"Processing patient: {pid}")
        h_list = [x.to(device) for x in data]  # h: list of modality tensors
        h_tensor = torch.cat(h_list, dim=1)  # shape: [n_patches, 13824]
        attention_weights = get_attention_from_model(model, h_list, device)  # shape: [n_patches]

        wrapped_model = MSAMILBagWrapper(model)
        wrapped_model.to(device).eval()

        explainer = shap.GradientExplainer(wrapped_model, [h_tensor])
        shap_values = explainer.shap_values([h_tensor])  # [n_patches, 13824, 2]
        shap_class1 = shap_values[1]  # shape: [n_patches, 13824]

        shap_wsi = compute_attention_weighted_shap(shap_class1, attention_weights) # shape: [13824]
        shap_wsi_modality = aggregate_shap_to_modalities(shap_wsi.reshape(1, -1)).flatten()  # shape: [9]
        all_shap_wsi.append(shap_wsi_modality)

        h_wsi = compute_attention_weighted_shap(h_tensor.cpu().numpy(), attention_weights)  # shape: [13824]
        input_feat = aggregate_shap_to_modalities(h_wsi.reshape(1, -1)).flatten()         # shape: [9]
        input_feat_list.append(input_feat)  # shape: [9]

        with torch.no_grad():
            logits = wrapped_model(h_tensor)
            baseline = logits[0, 1].item()

        save_subdir = os.path.join(save_dir, f"patient_{pid}")
        os.makedirs(save_subdir, exist_ok=True)
        

        plot_modality_bar(shap_wsi_modality, modality_names, save_path=os.path.join(save_subdir, f"shap_modality_bar.png"))
        plot_shap_cluster_heatmap(shap_class1, save_path=os.path.join(save_subdir, f"shap_cluster.png"))


        # 4. force plot（单个样本模态聚合解释）
        plt.figure()
        shap.force_plot(
            base_value=baseline,
            shap_values=shap_wsi_modality.astype(float),
            features=input_feat.astype(float),
            feature_names=modality_names,
            matplotlib=True,
            show=False
        )
        plt.savefig(os.path.join(save_subdir, f"shap_force.png"))
        plt.close()
        print(f"force_plot saved to {os.path.join(save_subdir, f'shap_force.png')}")


    shap_matrix = np.stack(all_shap_wsi, axis=0)  # shape: [n_patients, 9]
    X_shap = np.stack(input_feat_list, axis=0)

    shap_data = {
        "shap_values": shap_matrix,        # shape: (n_samples, n_modalities)
        "features": X_shap        # shape: (n_samples, n_modalities)
    }
    save_path = os.path.join(save_subdir, 'shap_outputs')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"shap_fold_{fold_idx}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(shap_data, f)

    return shap_matrix, X_shap  # 分别为 [n_samples, 9]

def shap_summary_plot(shap_matrix, X_shap, save_dir):
    modality_names = ["CD4", "CD8", "CD20", "CD56", "CD68", "CD163", "FOXP3", "PDL1", "HE"]
    expl = shap.Explanation(
        values=shap_matrix,
        data=X_shap,
        feature_names=modality_names
    )
    plt.figure()
    shap.plots.beeswarm(expl, show=False)
    # shap.summary_plot(expl, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"shap_beeswarm_all.png"))
    plt.close()

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    modality_names = np.array(modality_names)
    sorted_idx = np.argsort(mean_abs_shap)[::1]  # 降序
    plt.figure(figsize=(8, 5))
    plt.barh(modality_names[sorted_idx], mean_abs_shap[sorted_idx])
    plt.xlabel("Mean(|SHAP value|)", fontsize=16)
    plt.yticks(fontsize=16)  # y轴刻度字体
    plt.xticks(fontsize=16)  # x轴刻度字体
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"shap_bar_all.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.plots.bar(expl, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar_all_shaplib.png"))
    plt.close()


    plt.figure(figsize=(10, 6))
    sns.heatmap(shap_matrix, xticklabels=modality_names, cmap="coolwarm", center=0)
    plt.xlabel("Modality")
    plt.ylabel("Patient Index")
    plt.title("SHAP Heatmap Across Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"shap_heatmap_all.png"))
    plt.close()

    # plot_shap_cluster_heatmap(shap_matrix, save_path=os.path.join(save_dir, "shap_cluster_heatmap.png"))

    



def aggregate_shap_to_modalities(shap_values, modality_dim=1536, num_modalities=9, use_abs=False):
    """
    输入：
        shap_values: np.array, 形状为 (n_patches, 13824) - 单类别SHAP值
    输出：
        modality_shap: np.array, 形状为 (num_modalities,), 每个模态的平均绝对SHAP值
    """
    modality_shap = []
    for i in range(num_modalities):
        start = i * modality_dim
        end = (i + 1) * modality_dim
        modality_vals = shap_values[:, start:end]  # shape: (n_patches, modality_dim)
        if use_abs:
            val = np.mean(np.abs(modality_vals))
        else:
            val = np.mean(modality_vals)
        modality_shap.append(val)
    return np.array(modality_shap)  # (9,)



def compute_attention_weighted_features(h_tensor, attention_weights):
    """对所有patch特征进行attention加权聚合"""
    # h_tensor: [n_patches, 13824], attention_weights: [n_patches]
    return torch.sum(attention_weights.unsqueeze(1) * h_tensor, dim=0)

def aggregate_modalities(h_wsi):
    """按模态聚合特征：例如每个模态维度为 [1536]，共9种模态"""
    return h_wsi.view(-1, 1536).mean(dim=1)  # shape: [9]


if __name__ == "__main__":
    shap_path= "/local5/jzd/Code/MSAMIL-main/eval_results/EVAL_escc_gigapath_msamil_M9_s1_cv/shap_plots/patient_240/shap_outputs/shap_fold_2.pkl"  # 替换为实际的SHAP数据目录

    # all_shap_values = []
    # all_X_shap = []
    with open(shap_path, "rb") as f:
        shap_data = pickle.load(f)
        # all_shap_values.append(shap_data["shap_values"])
        # all_X_shap.append(shap_data["features"])
        shap_matrix = shap_data["shap_values"]  # shape: [n_samples, 9]
        X_shap = shap_data["features"]  # shape: [n_samples, 9]
        save_dir = "/local5/jzd/Code/MSAMIL-main/eval_results/EVAL_escc_gigapath_msamil_M9_s1_cv/shap_plots/patient_240/shap_outputs"  # 替换为实际的保存目录
        shap_summary_plot(shap_matrix, X_shap, save_dir)