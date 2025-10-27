import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import *


class MSDLM(nn.Module):
    def __init__(self, dropout = 0., n_classes=2, n_modalities=1, lo_dim = 32, mmhid=64, embed_dim=1024, method='mean'):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities
        self.n_classes = n_classes
        self.method = method

        lo_dims = [lo_dim for i in range(n_modalities)]
        self.fc1 = [nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.ReLU(), nn.Dropout(p=dropout), 
            nn.Linear(embed_dim // 4, lo_dim), nn.ReLU(), nn.Dropout(p=dropout)
        ) for i in range(n_modalities)]
        self.fc1 = nn.ModuleList(self.fc1)

        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.ReLU(), nn.Dropout(p=dropout), 
            nn.Linear(embed_dim // 4, lo_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Sequential(nn.Linear(embed_dim, lo_dim), nn.ReLU(), nn.Dropout(p=dropout))
        
        self.fusion = Fusion(lo_dims, mmhid, dropout)

        self.classifier = nn.Linear(mmhid, n_classes)

        fc = [nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L = mmhid, D = mmhid, dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

    def forward(self, h):
        all_input = []
        for modality_idx in range(self.n_modalities):
            modality_h = h[modality_idx]  # [n_patches, embed_dim]
            if self.embed_dim==32:
                input = modality_h
            else:
                input = self.fc1[modality_idx](modality_h)
            all_input.append(input)

        fused_features = self.fusion(*all_input)   # [n_patches, mmhid]

        logits = self.classifier(fused_features)   # [n_patches, n_classes]

        Y_prob = F.softmax(logits, dim=1)          # [n_patches, n_classes]

        if self.method == 'mean':
            Y_prob_p = Y_prob.sum(dim=0) / Y_prob.shape[0]
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]
        elif self.method == 'max':
            Y_prob_p, _ = Y_prob.max(dim=0)  # [n_classes]
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]
        elif self.method == 'LogSumExp':
            Y_prob_p = torch.logsumexp(Y_prob, dim=0) - torch.log(torch.tensor(Y_prob.shape[0], device=Y_prob.device, dtype=Y_prob.dtype))
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]
        elif self.method == 'attention':
            attn_weights = self.attention_net(fused_features)  # [n_patches, 1]
            attn_weights = torch.softmax(attn_weights, dim=0)
            Y_prob_p = (attn_weights * Y_prob).sum(dim=0)  # [n_classes]
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]
        elif self.method == 'topk':
            k = max(1, Y_prob.shape[0] // 10)  # 例如取前20%的patch
            topk_probs, _ = Y_prob.topk(k, dim=0)  # [k, n_classes]
            Y_prob_p = topk_probs.mean(dim=0)
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]
        elif self.method == 'classifier':
            bag_embedding = fused_features.mean(dim=0, keepdim=True)  # [1, mmhid]
            Y_prob_p = F.softmax(self.classifier(bag_embedding), dim=1).squeeze(0)
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]
        elif self.method == 'majority':
            patch_preds = torch.argmax(Y_prob, dim=1)                  # [n_patches]
            Y_hat_p = torch.mode(patch_preds, dim=0)[0].unsqueeze(0)   # [1]
            Y_prob_p = torch.bincount(patch_preds, minlength=Y_prob.shape[1]).float()
            Y_prob_p /= Y_prob_p.sum()
        elif self.method == 'logits_avg':
            avg_logits = logits.mean(dim=0)
            Y_prob_p = F.softmax(avg_logits, dim=0)
            Y_hat_p = torch.topk(Y_prob_p, 1, dim=0)[1]

        else:
            raise ValueError(f"无效的方法 '{self.method}'")

        results_dict = {}

        return logits, Y_prob_p, Y_hat_p, None, results_dict

