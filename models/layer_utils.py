import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from nystrom_attention import Nystromformer


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D), nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class Fusion(nn.Module):
    def __init__(self, lo_dims, mmhid, dropout_rate):
        super(Fusion, self).__init__()
        added_dim = int(np.sum(lo_dims))
        self.values = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for dim in lo_dims])
        self.attention_scores = nn.ModuleList([nn.Linear(added_dim, dim) for dim in lo_dims])
        self.outputs = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=dropout_rate)) for dim in lo_dims])

        self.encoder1 = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear((added_dim), mmhid * 2), nn.ReLU(),
                                      nn.Dropout(p=dropout_rate),
                                      nn.Linear(mmhid * 2, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, *inputs):
        cat_vec = torch.cat((inputs), dim=1)
        outputs = []
        for idx,input in enumerate(inputs):
            value = self.values[idx](input)
            attention = nn.Sigmoid()(self.attention_scores[idx](cat_vec))
            output = self.outputs[idx](attention * value)
            outputs.append(output)

        no_kronecker = torch.cat((outputs), dim=1)
        out = self.encoder1(no_kronecker)
        out = self.encoder2(out)
        return out


def Instance_Embedding_Net(dim_in=1024, dim_out=1024, dropout=0.5, arch='default'):
    if arch == 'default':
        f = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )
    elif arch == 'nonlinear':
        f = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )
    else:
        f = nn.Identity()

    return f


def Instance_Dependency_Learning_Net(feat_dim, dropout=0.5, arch='default'):
    if arch == 'default':
        s = nn.Identity()
    elif arch == 'sa-tf':
        # self-attention layer
        patch_encoder_layer = nn.TransformerEncoderLayer(
            feat_dim, 8, dim_feedforward=feat_dim, 
            dropout=dropout, activation='relu', batch_first=True
        )
        s = nn.TransformerEncoder(patch_encoder_layer, num_layers=1)
    elif arch == 'sa-nf':
        s = Nystromformer(
            dim=feat_dim, depth=1, heads=8,
            attn_dropout=dropout
        )
    else:
        s = nn.Identity()
    return s


class Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(Feat_Projecter, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x = [B, N, C] or [N, C]
        if len(x.shape) == 3:
            L1, L2, L3 = x.shape[0], x.shape[1], x.shape[2]
            x = x.view(-1, L3)
            x = self.projecter(x) # for BatchNorm
            x = x.view(L1, L2, -1)
        else:
            x = self.projecter(x)
        return x


class Gated_Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by 
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim, hid_dim, dropout=0.5):
        super(Gated_Attention_Pooling, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x, ret_raw_attn=False):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        emb = self.fc1(x) # [B, N, d']
        scr = self.score(x) # [B, N, d'] \in [0, 1]
        new_emb = emb.mul(scr)
        A_ = self.fc2(new_emb) # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1) # [B, 1, N]
        A = F.softmax(A_, dim=2) # [B, 1, N]
        out = torch.matmul(A, x).squeeze(1) # [B, 1, d]
        if ret_raw_attn:
            A_ = A_.squeeze(1) # [B, N]
            return out, A_
        else:
            A = A.squeeze(1) # [B, N]
            return out, A


class Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by 
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim=1024, hid_dim=512):
        super(Attention_Pooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x, ret_raw_attn=True):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        A_ = self.attention(x)  # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1)  # [B, 1, N]
        attn = F.softmax(A_, dim=2)  # [B, 1, N]
        out = torch.matmul(attn, x).squeeze(1)  # [B, 1, N] bmm [B, N, d] = [B, 1, d]
        if ret_raw_attn:
            A_ = A_.squeeze(1)
            return out, A_
        else:
            A = A.squeeze(1)
            return out, A


# DSMIL
from skimage import exposure

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        
        attentions = np.array(A.detach().cpu())
        attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))

        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B
        