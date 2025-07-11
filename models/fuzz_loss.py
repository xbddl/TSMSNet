import torch
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt


from models.check_nan import collect_non_nan



def get_mask(label,edge,edge_weight,change_weight,unchange_weight,rate):
    
    edge_mask = edge

    edge_mask = torch.where(edge_mask > 0, torch.ones_like(edge_mask), torch.zeros_like(edge_mask))

    
    change_mask = label[:,1,:,:]

    change_mask = change_mask.unsqueeze(1)

    change_mask_1 = torch.where(change_mask > 0, torch.ones_like(change_mask), torch.zeros_like(change_mask))

    
    change_mask = change_mask_1 - (edge_mask * change_mask_1)

    
    mask = edge_weight * edge_mask + change_weight * rate * change_mask + unchange_weight * (1 - edge_mask - change_mask)
    mask = torch.where(mask > 0, mask, torch.ones_like(mask))

    return mask

def test_ce_loss(pred,label):
    eps = 1e-10
    pred = F.softmax(pred, dim=1)

    pred_log = torch.log(pred + eps)

    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    loss = -1.0 * label * pred_log
    loss = loss.sum() / (label.sum())

    return loss

def test_local_mask_ce_loss(pred,label):
    eps = 1e-10
    pred = F.softmax(pred, dim=1)
    fuzz_weight = get_fuzz_weight_v2(pred)
    pred_log = torch.log(pred + eps)

    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    
    condition = fuzz_weight <= 0
    count = condition.sum().item()
    print(count)
    
    loss = -1.0 * label * pred_log * fuzz_weight
    loss = torch.mean(loss)

    return loss

def test_global_mask_ce_loss(pred,label,edge):
    edge_weight = 3
    change_weight = 2
    unchange_weight = 1
    rate = 1.0
    
    eps = 1e-10
    pred = F.softmax(pred, dim=1)

    pred_log = torch.log(pred + eps)
    

    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    
    mask = get_mask(label,edge,edge_weight,change_weight,unchange_weight,rate)
    
    loss = -1.0 * mask* label * pred_log
    loss = loss.sum() / (label.sum())

    return loss


def get_fuzz_weight_v2(pred):
    
    sqrt_2 = torch.sqrt(torch.tensor(2.0,dtype=pred.dtype,device=pred.device))
    a = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    
    diff = torch.abs(pred - a)
    squared_diff = torch.square(diff)
    squared_diff = squared_diff.sum(dim=1,keepdim=True)
    
    fuzz_degree = torch.sqrt(squared_diff)
    fuzz_degree = sqrt_2 * fuzz_degree
    
    #计算fuzz_degree/2 + 0.5
    fuzz_weight = fuzz_degree/2 + 0.5
    return fuzz_weight

def fuzz_loss_v2(pred,label,edge,precision,recall,epoch):
    
    # define global weights
    edge_weight = 3
    change_weight = 2
    unchange_weight = 1
    rate = precision / recall if epoch >= 150 else 1.0
    
    eps = 1e-10
    pred = F.softmax(pred, dim=1)
    
    pred_log = torch.log(pred + eps)
    
    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    
    mask = get_mask(label,edge,edge_weight,change_weight,unchange_weight,rate)
    
    fuzz_weight = get_fuzz_weight_v2(pred)
    
    loss = -1.0 * label * pred_log * mask * fuzz_weight
    loss = loss.sum() / (label.sum())
    
    return loss


def get_fuzz_weight_v1(pred):
    
    eps = 1e-10
    sqrt_2 = torch.sqrt(torch.tensor(2.0,dtype=pred.dtype,device=pred.device))
    a = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    
    diff = pred - a
    squared_diff = diff ** 2
    squared_diff = squared_diff.sum(dim=1,keepdim=True)

    fuzzy_degree = torch.sqrt(squared_diff+eps)
    fuzzy_degree = fuzzy_degree * sqrt_2
    average_fuzzy_degree = fuzzy_degree.mean()
    
    fuzz_weight = fuzzy_degree - average_fuzzy_degree
    fuzz_weight = 2 * torch.sigmoid(fuzz_weight)
    
    return fuzz_weight


def test_local_mask_ce_loss_v1(pred,label):
    eps = 1e-10
    
    pred = F.softmax(pred, dim=1)
    fuzz_weight = get_fuzz_weight_v1(pred)
    pred_log = torch.log(pred + eps)

    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    
    
    loss = -1.0 * label * pred_log * fuzz_weight
    loss = torch.mean(loss)

    return loss

def fuzz_loss_v1(pred,label,edge,precision,recall,epoch):
    
    # define global weights
    edge_weight = 2
    change_weight = 1
    unchange_weight = 1
    rate = precision/recall if epoch >150 else 1.0
    
    label = label.long()
    label = F.one_hot(label, num_classes=2).float().squeeze(1)
    label = label.permute(0, 3, 1, 2).contiguous()
    
    mask = get_mask(label,edge,edge_weight,change_weight,unchange_weight,rate)
    
    eps = 1e-10
    pred = F.softmax(pred, dim=1)
    pred_log = torch.log(pred+eps)

    fuzz_weight = get_fuzz_weight_v1(pred)
    
    loss = -1.0 * label * pred_log * mask * fuzz_weight
    loss = loss.sum() / (label.sum())

    return loss


def calculate_nan_not_nan(tensor,mode=False):
    nan_mask = torch.isnan(tensor)
    num_nan = (nan_mask == 1).sum().item()
    
    not_nan_mask = ~nan_mask
    num_not_nan = (not_nan_mask == 1).sum().item()
    if mode == True:
        return num_nan,num_not_nan,not_nan_mask
    else:
        return num_nan,num_not_nan

def calculate_pred_(pred):
    count_zero = torch.sum(pred == 0).item()
    count_one = torch.sum(pred == 1).item()
    count_nans,count_not_nans = calculate_nan_not_nan(pred)
    count_negative = torch.sum(pred < 0).item()
    count_positive = torch.sum(pred > 0).item()
    return count_zero,count_one,count_nans,count_not_nans,count_negative,count_positive