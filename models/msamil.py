import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import *
# from .dsmil import DSMIL


class MSAMIL(nn.Module):
    def __init__(self, mil_method='abmil', gate = True, dropout = 0., n_classes=2, n_modalities=1, lo_dim = 32, mmhid=64, embed_dim=1024):
        nn.Module.__init__(self)
        self.method = mil_method
        self.n_classes = n_classes
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim

        self.fc = [nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.ReLU(), nn.Dropout(p=dropout), 
            nn.Linear(embed_dim // 4, lo_dim), nn.ReLU(), nn.Dropout(p=dropout)
        ) for i in range(n_modalities)]
        self.fc = nn.ModuleList(self.fc)
        lo_dims = [lo_dim for i in range(n_modalities)]
        self.fusion = Fusion(lo_dims, mmhid, dropout)

        fc = [nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L = mmhid, D = mmhid, dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        bag_classifiers = [nn.Linear(mmhid, 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
            

    def forward(self, h, return_features=False, attention_only=False):
        all_input = []
        for modality_idx in range(self.n_modalities):
            modality_h = h[modality_idx]
            input = self.fc[modality_idx](modality_h)
            all_input.append(input)
        fused_features = self.fusion(*all_input)   # [n_patches, mmhid]

        A, h = self.attention_net(fused_features)    
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)                                              # [2, mmhid]
        logits = torch.empty(1, self.n_classes).float().to(M.device)    # [1, n_classes]
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]                       # [1, 1]
        Y_prob = F.softmax(logits, dim = 1)                             # [1, n_classes]
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
