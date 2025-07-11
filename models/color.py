import numpy as np

def compare_gt_pred(gt,pred):
    w = pred.shape[1]
    h = pred.shape[0]
    result = np.zeros((h,w,3),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if gt[i,j,0] != 0 and pred[i,j,0] == 0:
                result[i,j,0] = 0
                result[i,j,1] = 255
                result[i,j,2] = 0
            elif gt[i,j,0] == 0 and pred[i,j,0] == 255:
                result[i,j,0] = 255
                result[i,j,1] = 0
                result[i,j,2] = 0
            elif gt[i,j,0] != 0 and pred[i,j,0] == 255:
                result[i,j,0] = 255
                result[i,j,1] = 255
                result[i,j,2] = 255
                
    return result