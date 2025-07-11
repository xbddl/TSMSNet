import torch.nn as nn
import torch.nn.functional as F
import torch
class cd_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
    def forward(self,input,target):
        input = F.sigmoid(input)
        target = target.unsqueeze(1)  # [:, :1, :, :]
        target = F.interpolate(target, size=input.shape[2:], mode='nearest')
        bce_loss = self.bce_loss(input, target)

        smooth = 1e-10
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        dic_loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        return dic_loss + bce_loss


