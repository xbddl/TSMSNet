
from .metrics import FocalLoss, dice_loss
import torch.nn as nn



class hybrid_loss(nn.Module):
    def __init__(self):
        super(hybrid_loss, self).__init__()
        self.focal = FocalLoss(gamma=0, alpha=None)
        self.dice_loss = dice_loss()
    def forward(self, predictions, target):
        loss = 0
        for prediction in predictions:
            bce = self.focal(prediction, target)
            dice = self.dice_loss(prediction, target)
            loss += bce + dice

        return loss