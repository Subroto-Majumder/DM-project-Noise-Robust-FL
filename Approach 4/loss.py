import torch.nn as nn
import torch
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        super(CELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.ce(outputs, targets.long())

class RobustLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        """
        Implements the Active-Passive Loss.
        """
        super(RobustLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def forward(self, outputs, targets):
        # Active Loss: Normalized Cross Entropy
        log_probs = F.log_softmax(outputs, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(outputs.device)
        nce = -1 * (torch.sum(log_probs * one_hot, dim=1)) / (-log_probs.sum(dim=1) + self.epsilon)
        nce = nce.mean()

        # Passive Loss: Normalized Reverse Cross Entropy
        pred = F.softmax(outputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(targets, self.num_classes).float().to(outputs.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        rce = rce.mean()

        total_loss = self.alpha * nce + self.beta * rce
        return total_loss

