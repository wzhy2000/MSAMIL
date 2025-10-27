import torch
import torch.nn as nn

from .layer_utils import *


class DSMIL(nn.Module):
    def __init__(self, embed_dim, n_classes=2, dropout=0.25):
        super(DSMIL, self).__init__()
        self.embed_dim = embed_dim
        
        self.i_classifier = FCLayer(in_size=embed_dim, out_size=n_classes)

        self.b_classifier = BClassifier(
            input_size=embed_dim,
            output_class=n_classes,
            dropout_v=dropout,
            nonlinear=True
        )

    def forward(self, h, attention_only=False):
        h = h.reshape(-1, self.embed_dim)

        feats, classes = self.i_classifier(h)

        prediction_bag, A, B = self.b_classifier(feats, classes)

        logit = prediction_bag  # [1, C]
        Y_prob = torch.softmax(logit, dim=-1)  # [1, 2]
        Y_hat = torch.argmax(logit, dim=-1)    # [1]

        # attn = A.detach()  # [N, C]
        if attention_only:
            return A
        else:
            return logit, Y_prob, Y_hat, A, None
