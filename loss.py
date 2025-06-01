import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

def sobel(img):
    """ Assumes img is a torch tensor of shape [B, 1, H, W] or [B, C, H, W] """
    device = img.device
    channels = img.shape[1]
    sobel_x = torch.tensor([[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]], dtype=img.dtype, device=device)
    sobel_y = torch.tensor([[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]], dtype=img.dtype, device=device)
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=channels)
    return grad_x, grad_y

def L_depth(y, y_pred):
    return F.l1_loss(y, y_pred)
    
def L_grad(y, y_pred):
    true_x_grad, true_y_grad = sobel(y)
    pred_x_grad, pred_y_grad = sobel(y_pred)
    g_x = true_x_grad - pred_x_grad
    g_y = true_y_grad - pred_y_grad
    """ 
        Since the loss is 1/n * sum(|g_x| + |g_y|) (1) and L1 loss is of the form 1/n * sum(|z_1-z_2|),
        we can plug |g_x| and -|g_y| into z_1 and z_2 respectively to obtain (1)
    """
    return F.l1_loss(abs(g_x), -abs(g_y))

def L_ssim(y, y_pred):
    return (1-ssim(y, y_pred))/2

class Loss:
    def __init__(self, Lambda):
        self.Lambda = Lambda
        
    def __call__(self, y, y_pred):
        return self.Lambda*L_depth(y, y_pred) + L_grad(y, y_pred) + L_ssim(y, y_pred)
