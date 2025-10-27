import torch
import torch.nn as nn
import torchvision.ops.focal_loss as focal_loss
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C] logits or probabilities
        targets: [N] int64 labels
        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
        weight = (1 - probs) ** self.gamma
        focal = -self.alpha * weight * targets_one_hot * log_probs
        loss = focal.sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


