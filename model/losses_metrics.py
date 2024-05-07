import torch
import segmentation_models_pytorch as smp

DiceLoss    = smp.losses.DiceLoss(mode='binary')
BCELoss     = torch.nn.BCEWithLogitsLoss()

def dice_coef(y_pred, y_true, thr=0.5, epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = torch.mean((2.0 * intersection) / (union + epsilon)) 
    return dice

def iou_coef(y_pred, y_true, thr=0.5, epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    intersection = torch.sum(y_true*y_pred)
    union = torch.sum(y_true + y_pred - y_true*y_pred)
    iou = torch.mean(((intersection+epsilon)/(union+epsilon)))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*DiceLoss(y_pred, y_true) 