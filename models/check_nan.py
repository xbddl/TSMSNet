import torch
import pdb

def collect_non_nan(pred_0,pred_1,pred_2,pred,label,edge):
    non_nan_pred_0 = []
    non_nan_pred_1 = []
    non_nan_pred_2 = []
    non_nan_pred = []
    non_nan_label = []
    non_nan_edge = []
    for i in range(pred.size(0)):
        pred_0_i = pred_0[i]
        pred_1_i = pred_1[i]
        pred_2_i = pred_2[i]
        pred_i = pred[i]
        label_i = label[i]
        edge_i = edge[i]
        if torch.isnan(pred_i).any():
            continue
        non_nan_pred_0.append(pred_0_i)
        non_nan_pred_1.append(pred_1_i)
        non_nan_pred_2.append(pred_2_i)
        non_nan_pred.append(pred_i)
        non_nan_label.append(label_i)
        non_nan_edge.append(edge_i)
        
    if non_nan_pred:
        non_nan_pred_0 = torch.stack(non_nan_pred_0, dim=0)
        non_nan_pred_1 = torch.stack(non_nan_pred_1, dim=0)
        non_nan_pred_2 = torch.stack(non_nan_pred_2, dim=0)
        non_nan_pred = torch.stack(non_nan_pred, dim=0)
        non_nan_label = torch.stack(non_nan_label, dim=0)
        non_nan_edge = torch.stack(non_nan_edge, dim=0)
    else:
        non_nan_pred_0 = torch.tensor(0,*pred_0.size()[1:], dtype=torch.float32)
        non_nan_pred_1 = torch.tensor(0,*pred_1.size()[1:], dtype=torch.float32)
        non_nan_pred_2 = torch.tensor(0,*pred_2.size()[1:], dtype=torch.float32)
        non_nan_pred = torch.tensor(0,*pred.size()[1:], dtype=torch.float32)
        non_nan_label = torch.tensor(0,*label.size()[1:], dtype=torch.float32)
        non_nan_edge = torch.tensor(0,*edge.size()[1:], dtype=torch.float32)
        
    return non_nan_pred_0,non_nan_pred_1,non_nan_pred_2,non_nan_pred,non_nan_label,non_nan_edge

def check_is_all_nan(pred_0,pred_1,pred_2,pred,label,edge):
    """_summary_

    Args:
        pred_0 (_type_): _description_
        pred_1 (_type_): _description_
        pred_2 (_type_): _description_
        pred (_type_): _description_
        label (_type_): _description_
        edge (_type_): _description_

    Returns:
        pred_0 (_type_): _description_
        pred_1 (_type_): _description_
        pred_2 (_type_): _description_
        pred (_type_): _description_
        label (_type_): _description_
        edge (_type_): _description_
    """
    pred_0,pred_1,pred_2,pred,label,edge = collect_non_nan(pred_0,pred_1,pred_2,pred,label,edge)
    is_zero = torch.eq(pred,0)
    if torch.all(is_zero):
        pdb.set_trace()
    return pred_0, pred_1, pred_2, pred, label, edge

