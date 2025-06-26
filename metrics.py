import torch
import torch.nn.functional as F 

def rel(y, y_hat):
    diff = abs(y-y_hat)
    relative_errors = diff/y
    avg_rel_err = torch.mean(relative_errors)
    return avg_rel_err

def rmse(y, y_hat):
    error = torch.sqrt(F.mse_loss(y, y_hat))
    return error

def log_10(y, y_hat):
    error = F.l1_loss(torch.log10(y), torch.log10(torch.clip(y_hat, 0.4, 10)))
    return error

def lambda_i(y, y_hat, i):
    thr = 1.25**i
    max_map = torch.maximum(y/y_hat, y_hat/y)
    truth_map = torch.where(max_map < thr, 1.0, 0.0)
    percent = torch.mean(truth_map)
    return percent
    