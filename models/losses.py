import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.fuzz_loss import test_local_mask_ce_loss_v1 as fuzz_loss

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
    
def multi_cross_entropy(pred_0, pred_1, pred_2, input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    """
    ce_loss = cross_entropy
    loss = ce_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)
    loss1 = ce_loss(pred_0, target, weight=weight, reduction=reduction, ignore_index=ignore_index)
    target = F.interpolate(target, size=pred_1.shape[2:],scale_factor=None, mode='nearest')
    loss2 = ce_loss(pred_1, target, weight=weight, reduction=reduction, ignore_index=ignore_index)
    target = F.interpolate(target, size=pred_2.shape[2:],scale_factor=None, mode='nearest')
    loss3 = ce_loss(pred_2, target, weight=weight, reduction=reduction, ignore_index=ignore_index)

    
    return loss + loss1 + loss2 + loss3
def deep_supervised_ce(pred_0, pred_1, pred_2, input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    """
    return multi_cross_entropy(pred_0, pred_1, pred_2, input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)


def feature_affinity_loss(h_feature,l_feature, weight=None, reduction='mean', ignore_index=255):
    """
    h_feature: torch.Tensor, N*C1*h1*w1, high level feature
    l_feature: torch.Tensor, N*C2*h2*w2, low level feature
    """
    h_feature = F.interpolate(h_feature, size=l_feature.shape[2:],scale_factor=None, mode='bilinear', align_corners=True)
    loss = F.mse_loss(h_feature, l_feature, reduction=reduction, weight=weight, ignore_index=ignore_index)
    return loss
    
def edge_aware_loss(pred, label, edge, T=0.75, a=0.1, extra_weight=1.0):
    """
    set cross entropy weight of edge region  to 2 , others to 1
    Args:
        pred (_type_): _description_
        label (_type_): _description_
        edge (_type_): _description_
        extra_weight (float, optional): _description_. Defaults to 1.0.
    """
    weight = edge * extra_weight + 1.0
    weight = weight.unsqueeze(1)
    pred = F.log_softmax(pred, dim=1)
    
    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    for i in label:
        count_unch = torch.sum(i[0] == 1).item()
        count_ch = torch.sum(i[1] == 1).item()
        all = count_ch + count_unch
        r_unch = count_unch/all
        r_ch = count_ch/all
        if r_unch > T:
            r_unch = r_unch - a
            r_ch = r_ch + a
        elif r_unch < 1 - T:
            r_unch = r_unch + a
            r_ch = r_ch - a
            
        i[0] = r_ch * i[0]
        i[1] = r_unch * i[1]
    
    loss = -1.0 * label * pred
    loss = loss * weight
    
    return loss.mean()

def edge_aware_loss_v2(pred, label, edge):
    """
    a*ce + b*edge_loss
    Args:
        pred (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
        
    """
    edge_mask = edge
    edge_mask = edge_mask.unsqueeze(1)
    pred = F.log_softmax(pred, dim=1)
    
    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    
    edge_loss = -1.0 * label * pred * edge_mask
    edge_loss = edge_loss.mean()
    
    ce = -1.0 * label * pred
    ce = ce.mean()
    
    return edge_loss + ce

    
def deep_supervised_fuzz_loss(pred_0, pred_1, pred_2, input, target, edge=None, precision=None, recall=None, epoch=None, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    """
    ce_loss = cross_entropy
    loss = fuzz_loss(input,target)
    #print(loss)
    loss_0 = ce_loss(pred_0, target)
    #print(loss_0)
    target = F.interpolate(target, size=pred_1.shape[2:],scale_factor=None, mode='nearest')
    loss_1 = ce_loss(pred_1, target)
    
    target = F.interpolate(target, size=pred_2.shape[2:],scale_factor=None, mode='nearest')
    loss_2 = ce_loss(pred_2, target)
    
    return loss + loss_0 + loss_1 + loss_2

def NR_Dice_CE_Loss(pred, label):
    epsilon = 1e-8
    gamma = 1.5
    
    
    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    pred1 = F.softmax(pred, dim=1)
    
    l1 = abs(pred1 - label)**gamma
    y_pred_sqsum = torch.sum((pred1*pred1))
    y_true_sqsum = torch.sum((label*label))
    l1_sum = torch.sum(l1)
    score = (l1_sum) / (y_pred_sqsum + y_true_sqsum + epsilon)
    nr_dice_loss = score.mean()
    
    pred = F.log_softmax(pred, dim=1)
    
    ce_loss = -1.0 * label * pred
    ce_loss = ce_loss.mean()
    
    return nr_dice_loss + ce_loss

def deep_supervised_NR_Dice_loss(pred_0, pred_1, pred_2, input, target, weight=None, reduction='mean', ignore_index=255):
    loss = NR_Dice_CE_Loss(input, target)
    loss_0 = NR_Dice_CE_Loss(pred_0, target)
    target = F.interpolate(target, size=pred_1.shape[2:],scale_factor=None, mode='nearest')
    
    loss_1 = NR_Dice_CE_Loss(pred_1, target)
    target = F.interpolate(target, size=pred_2.shape[2:],scale_factor=None, mode='nearest')
    
    loss_2 = NR_Dice_CE_Loss(pred_2, target)
    
    return loss + loss_0 + loss_1 + loss_2
    
def deep_supervised_edge_loss(pred_0, pred_1, pred_2, input, target, edge, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    """
    loss = edge_aware_loss(input, target, edge)
    loss_0 = edge_aware_loss(pred_0, target, edge)
    target = F.interpolate(target, size=pred_1.shape[2:],scale_factor=None, mode='nearest')
    edge = F.interpolate(edge, size=pred_1.shape[2:],scale_factor=None, mode='nearest')
    loss_1 = edge_aware_loss(pred_1, target, edge)
    target = F.interpolate(target, size=pred_2.shape[2:],scale_factor=None, mode='nearest')
    edge = F.interpolate(edge, size=pred_2.shape[2:],scale_factor=None, mode='nearest')
    loss_2 = edge_aware_loss(pred_2, target, edge)
    
    return loss + loss_0 + loss_1 + loss_2


    

def Focal(logits, true, eps=1e-7):
    """
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)




class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        # batch equal to True means views all batch images as an entity and calculate loss
        # batch equal to False means calculate loss of every single image in batch and get their mean
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_focal_loss(nn.Module):

    def __init__(self):
        super(dice_focal_loss, self).__init__()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels)
        foclaloss = self.focal_loss(scores.clone(), labels)
        return [diceloss, foclaloss]


def FCCDN_loss_without_seg(scores, labels):
    # scores = change_pred
    # labels = binary_cd_labels
    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
    # if len(scores.shape) > 3:
    #     scores = scores.squeeze(1)
    # if len(labels.shape) > 3:
    #     labels = labels.squeeze(1)
    """ for binary change detection task"""
    criterion_change = dice_focal_loss()

    # change loss
    loss_change = criterion_change(scores[0], labels[0])
    loss_seg1 = criterion_change(scores[1], labels[1])
    loss_seg2 = criterion_change(scores[2], labels[1])

    for i in range(len(loss_change)):
        loss_change[i] += 0.2 * (loss_seg1[i] + loss_seg2[i])

    return loss_change
