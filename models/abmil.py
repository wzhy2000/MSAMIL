import torch
import torch.nn as nn

from .layer_utils import *


class DeepMIL(nn.Module):
    def __init__(self, embed_dim, dim_hid=1024, n_classes=2, use_feat_proj=True, dropout=0.5, 
        pooling='mean', pred_head='default'):
        super(DeepMIL, self).__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']

        self.embed_dim = embed_dim

        if use_feat_proj:
            self.feat_proj = Feat_Projecter(embed_dim, embed_dim)
        else:
            self.feat_proj = None
        
        if pooling == 'gated_attention':
            self.sigma = Gated_Attention_Pooling(embed_dim, dim_hid, dropout=dropout)
        elif pooling == 'attention':
            self.sigma = Attention_Pooling(embed_dim, dim_hid)
        else:
            self.sigma = pooling
        
        if pred_head == 'default':
            self.g = nn.Linear(embed_dim, n_classes)

    def forward(self, h, ret_with_attn=False):
        h = h.reshape(1, -1, self.embed_dim)

        assert h.shape[0] == 1
        if self.feat_proj is not None:
            h = self.feat_proj(h)

        if self.sigma == 'mean':
            X_vec = torch.mean(h, dim=1)
        elif self.sigma == 'max':
            X_vec, _ = torch.max(h, dim=1)
        else:
            X_vec, raw_attn = self.sigma(h)

        logit = self.g(X_vec)
        Y_prob = torch.softmax(logit, dim=-1)
        Y_hat = torch.argmax(logit, dim=-1)

        if ret_with_attn:
            attn = raw_attn.detach()
            return logit, Y_prob, Y_hat, attn, None
        else:
            return logit, Y_prob, Y_hat, None, None