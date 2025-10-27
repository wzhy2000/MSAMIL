import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .layer_utils import *
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn.detach()
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, embed_dim, dim_hid=512):
        super(TransMIL, self).__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        if self.embed_dim==32:
            dim_hid=32
        self.pos_layer = PPEG(dim=dim_hid)
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, dim_hid), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hid))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=dim_hid)
        self.layer2 = TransLayer(dim=dim_hid)
        self.norm = nn.LayerNorm(dim_hid)
        self._fc2 = nn.Linear(dim_hid, self.n_classes)

    def forward(self, h, attention=False):
        h = h.reshape(1, -1, self.embed_dim)
        if self.embed_dim==32:
            h = h
        else:
            h = self._fc1(h)

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        n1 = h.shape[1]

        h = self.layer1(h)

        h = self.pos_layer(h, _H, _W)
            
        if attention is not None:
            h, attn = self.layer2(h, return_attn=True)
            if add_length == 0:
                attn = attn[:, :, -n1, (-n1+1):]
            else:
                attn = attn[:, :, -n1, (-n1+1):(-n1+1+H)]
            attn = attn.mean(1).detach()
            assert attn.shape[1] == H
        else:
            h = self.layer2(h)
            attn = None

        h = self.norm(h)[:,0]

        logits = self._fc2(h)

        Y_prob = torch.softmax(logits, dim=-1)
        Y_hat = torch.argmax(logits, dim=-1)

        
        return logits, Y_prob, Y_hat, None, None
