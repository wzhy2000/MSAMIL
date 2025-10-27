import torch
import torch.nn as nn
import torch.nn.functional as F


class MIL_fc(nn.Module):
    def __init__(self, dropout = 0., n_classes = 2, top_k=1, embed_dim=1024, dim_hid=512):
        super().__init__()
        assert n_classes == 2
        fc = [nn.Linear(embed_dim, dim_hid), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        if embed_dim==32:
            self.classifier = nn.Linear(32, n_classes)
        else:
            self.classifier = nn.Linear(dim_hid, n_classes)
        self.top_k=top_k
        self.embed_dim = embed_dim

    def forward(self, h, return_features=False):
        h = h.reshape(-1, self.embed_dim)
        if self.embed_dim==32:
            h = h
        else:
            h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        y_probs = F.softmax(logits, dim = 1)

        avg_prob = y_probs.mean(dim=0)
        Y_hat = avg_prob.argmax(dim=0, keepdim=True)
        Y_prob = F.softmax(avg_prob, dim = 0) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=logits)
            results_dict.update({'features': top_features})
        return logits, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, dropout = 0., n_classes = 2, top_k=1, embed_dim=1024, dim_hid=512):
        super().__init__()
        assert n_classes > 2
        fc = [nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        if embed_dim==32:
            self.classifier = nn.Linear(32, n_classes)
        else:
            self.classifier = nn.Linear(dim_hid, n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        h = h.reshape(-1, self.embed_dim)
        if self.embed_dim==32:
            h = h
        else:
            h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
