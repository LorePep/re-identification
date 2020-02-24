import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
