import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.metrics import *
from scipy.stats import sem
from itertools import cycle, product
from sklearn.metrics import confusion_matrix
# from proplot import rc


# rc['tick.labelsize'] = 15
# rc["axes.labelsize"] = 17
# rc["axes.labelweight"] = "light"
# rc["tick.labelweight"] = "bold"

plt.rcParams.update({'font.size': 15})


def plot_k_roc(path, folds, marker='', type='mean', save_fig=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    file_names = [f'{path}/fold_{fold}.csv' for fold in range(folds)]

    if type == 'mean':
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for file_name in file_names:
            df = pd.read_csv(file_name)
            df['Second_Probability'] = df['p_1']
            fpr, tpr, thresholds = roc_curve(df['Y'], df['Second_Probability'])
            roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.plot(mean_fpr, mean_tpr, color="b", label=r"ROC (AUC %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc), lw=2, alpha=0.8)
        # print("mean_fpr.shape: ", mean_fpr.shape)
        # print("tprs_lower.shape: ", tprs_lower.shape)
        # print("tprs_upper: ", tprs_upper.shape)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, where=np.ones_like(mean_fpr, dtype=bool), color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    else:
        combined_df = pd.DataFrame()
        for file_name in file_names:
            df = pd.read_csv(file_name)
            df['Second_Probability'] = df['p_1']
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        fpr, tpr, thresholds = roc_curve(combined_df['Y'], combined_df['Second_Probability'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="b", label=r"ROC (AUC %0.3f)" % (roc_auc), lw=2, alpha=0.8)
        
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    plt.title(f"{marker}", font={'size':20})
    ax.legend(loc="lower right")
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    if save_fig:
        os.makedirs(f'{path}/roc_prc', exist_ok=True)
        plt.savefig(f'{path}/roc_prc/ROC.png')
    else:
        plt.show()
    plt.close()


def plot_k_prc(path, folds, marker='', type='mean', save_fig=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    file_names = [f'{path}/fold_{fold}.csv' for fold in range(folds)]
    
    if type == 'mean':
        precisions = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)
        for file_name in file_names:
            df = pd.read_csv(file_name)
            df['Second_Probability'] = df['p_1']
            precision, recall, threshold = metrics.precision_recall_curve(df['Y'], df['Second_Probability'])
            pr_auc = np.trapz(precision[::-1], recall[::-1])
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_precision)
            aucs.append(pr_auc)
        mean_precision = np.mean(precisions, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)

        ax.plot(mean_recall, mean_precision, color="b", label=r"PRC (AUC %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc), lw=2, alpha=0.8)
        # print("mean_recall.shape: ", mean_recall.shape)
        # print("precisions_lower.shape: ", precisions_lower.shape)
        # print("precisions_upper: ", precisions_upper.shape)
        ax.fill_between(mean_recall, precisions_lower, precisions_upper, where=np.ones_like(mean_recall, dtype=bool), color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    else:
        file_names = [f'{path}/fold_{fold}.csv' for fold in range(folds)]
        combined_df = pd.DataFrame()
        for file_name in file_names:
            df = pd.read_csv(file_name)
            df['Second_Probability'] = df['p_1']
            # print("combined_df.shape: ", combined_df.shape)
            # print("df.shape: ", df.shape)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        precision, recall, _ = metrics.precision_recall_curve(combined_df['Y'], combined_df['Second_Probability'])
        pr_auc = np.trapz(precision[::-1], recall[::-1])
        ax.plot(recall,precision, color="b", label=r"PRC (AUC %0.3f)" % (pr_auc), lw=2, alpha=0.8)

    ax.set(
        xlabel="Recall",
        ylabel="Precisions",
    )
    plt.title(f"{marker}", font={'size':20})
    ax.legend(loc="lower right")
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    if save_fig:
        os.makedirs(f'{path}/roc_prc', exist_ok=True)
        plt.savefig(f'{path}/roc_prc/PRC.png')
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, target_names,
                          path=None,
                          title='ConfusionMatrix',
                          cmap=None,
                          normalize=False, 
                          save_fig=True):
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(False)
    plt.title(f'{title}', font={'size':20})
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 1.5 if normalize else (cm.max() + cm.min()) / 2
    if len(target_names) < 10:
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, f'{cm[i, j]:.4f}', horizontalalignment='center',
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, f'{cm[i, j]}', horizontalalignment='center',
                         color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('\nPredicted Label')
    plt.ylabel('True Label')
    plt.tight_layout(h_pad=2.5)
    if save_fig:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_k_confusion_matrix(path, folds, save_fig=True):
    file_names = [f'{path}/fold_{fold}.csv' for fold in range(folds)]
    all_conf_matrix = np.zeros((2, 2), dtype = int) 
    for file_name in file_names:
        df = pd.read_csv(file_name)
        conf_matrix = confusion_matrix(df['Y'], df['Y_hat'])
        all_conf_matrix += conf_matrix
    TP = all_conf_matrix[1][1]
    FP = all_conf_matrix[0][1]
    FN = all_conf_matrix[1][0]
    TN = all_conf_matrix[0][0]
    cm = np.array([[TN, FP], [FN, TP]])

    target_names = ['No relapse', 'Relapse']
    os.makedirs(f'{path}/confusion_matrix', exist_ok=True)
    plot_confusion_matrix(cm, target_names, path=f'{path}/confusion_matrix/confusionmatrix.png',
                          title='confusion matrix',
                          cmap=plt.cm.Blues,
                          normalize=False,
                          save_fig=save_fig)
    
    plot_confusion_matrix(cm, target_names, path=f'{path}/confusion_matrix/confusionmatrix_radio.png',
                          title='confusion matrix',
                          cmap=plt.cm.Blues,
                          normalize=True,
                          save_fig=save_fig)


def plot_multiple_roc(path_list, marker_list, k=5, type='all', save_fig=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    mean_fpr = np.linspace(0, 1, 100)

    for i, (path, marker) in enumerate(zip(path_list, marker_list)):
        file_names = [f'{path}/fold_{j}.csv' for j in range(k)]

        if type == 'mean':
            tprs = []
            aucs = []
            for file_name in file_names:
                df = pd.read_csv(file_name)
                df['Second_Probability'] = df['p_1']
                fpr, tpr, thresholds = roc_curve(df['Y'], df['Second_Probability'])
                roc_auc = auc(fpr, tpr)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            color = 'black' if i == 0 else None
            ax.plot(mean_fpr, mean_tpr, label=f"{marker} (AUC = {mean_auc:.3f})", lw=2, color=color)

        else:
            combined_df = pd.DataFrame()
            for file_name in file_names:
                df = pd.read_csv(file_name)
                df['Second_Probability'] = df['p_1']
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            fpr, tpr, thresholds = roc_curve(combined_df['Y'], combined_df['Second_Probability'])
            roc_auc = auc(fpr, tpr)
            color = 'black' if i == 0 else None
            ax.plot(fpr, tpr, label=f"{marker} (AUC = {roc_auc:.3f})", lw=2, color=color)

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=20)
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    if save_fig:
        plt.savefig(f'{path_list[0]}/roc_combined.png')
        print(f'Saved ROC curve to {path_list[0]}/roc_combined.png')
    else:
        plt.show()
    plt.close()


def plot_multiple_prc(path_list, marker_list, k=5, type='all', save_fig=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    mean_recall = np.linspace(0, 1, 100)

    for i, (path, marker) in enumerate(zip(path_list, marker_list)):
        file_names = [f'{path}/fold_{j}.csv' for j in range(k)]
        if type == 'mean':
            precisions = []
            aucs = []
            for file_name in file_names:
                df = pd.read_csv(file_name)
                df['Second_Probability'] = df['p_1']
                precision, recall, threshold = metrics.precision_recall_curve(df['Y'], df['Second_Probability'])
                pr_auc = np.trapz(precision[::-1], recall[::-1])
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                precisions.append(interp_precision)
                aucs.append(pr_auc)
            mean_precision = np.mean(precisions, axis=0)
            mean_auc = np.mean(aucs)
            color = 'black' if i == 0 else None
            ax.plot(mean_recall, mean_precision, label=f"{marker} (AUC = {mean_auc:.3f})", lw=2, color=color)
            
        else:
            combined_df = pd.DataFrame()
            for file_name in file_names:
                df = pd.read_csv(file_name)
                df['Second_Probability'] = df['p_1']
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            precision, recall, _ = metrics.precision_recall_curve(combined_df['Y'], combined_df['Second_Probability'])
            pr_auc = np.trapz(precision[::-1], recall[::-1])
            color = 'black' if i == 0 else None
            ax.plot(recall, precision, label=f"{marker} (AUC = {pr_auc:.3f})", lw=2, color=color)
    
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
    )
    # ax.set_title("Precision-Recall Curves")
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)    
    ax.legend(loc="lower left", fontsize=20)

    ax.plot([0, 1], [1, 0], color='black', lw=2, linestyle='--')

    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    if save_fig:
        plt.savefig(f'{path_list[0]}/prc_combined.png')
    else:
        plt.show()
    plt.close()


def main():
    path_list = ['../eval_results(SOTA)/EVAL_escc_gigapath_msamil_M9_s1_cv',
                 '../eval_results(SOTA)/EVAL_escc_gigapath_msdlm_M9_s1_cv',
                 '../eval_results(SOTA)/EVAL_escc_gigapath_abmil_M9_s1_cv',
                 '../eval_results(SOTA)/EVAL_escc_gigapath_transmil_M9_s1_cv',
    ]
    marker_list = ["MSAMIL", "MSDLM", "ABMIL", "TransMIL"]
    plot_multiple_roc(path_list, marker_list)

    plot_multiple_prc(path_list, marker_list)

if __name__ == '__main__':
    main()