import torch
from model import Model
from metrics import *
from dataSetup import NYUDataSet
from tqdm import tqdm

PATH_TO_MODEL = "Test_6_17_2025_1\\model_weights.pth"

model = Model()
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.cuda()

test_dataset = NYUDataSet("nyu_data\\data\\nyu2_test.csv", "nyu_data")

rel_total = 0
rmse_total = 0
log_total = 0
lambda_1_total = 0
lambda_2_total = 0
lambda_3_total = 0

model.eval()

with torch.no_grad():
    for i in tqdm(range(len(test_dataset))):
        x, y = test_dataset[i]
        x = x.float().unsqueeze(0).cuda()
        y = y.float().unsqueeze(0).cuda()/1000
        
        y_pred = model.forward(x)*10
        y_pred = F.interpolate(y_pred, scale_factor=2)
        
        rel_total += rel(y, y_pred)
        rmse_total += rmse(y, y_pred)
        log_total += log_10(y, y_pred)
        lambda_1_total += lambda_i(y, y_pred, 1)
        lambda_2_total += lambda_i(y, y_pred, 2)
        lambda_3_total += lambda_i(y, y_pred, 3)
        
    rel_avg = rel_total / len(test_dataset)
    rmse_avg = rmse_total / len(test_dataset)
    log_avg = log_total / len(test_dataset)
    lambda_1_avg = lambda_1_total / len(test_dataset)
    lambda_2_avg = lambda_2_total / len(test_dataset)
    lambda_3_avg = lambda_3_total / len(test_dataset)

print(f"average relative error: {rel_avg:.3f}")
print(f"RMSE: {rmse_avg:.3f}")
print(f"Log_10 error: {log_avg:.3f}")
print(f"lambda_1 ratio: {lambda_1_avg:.3f}")
print(f"lambda_2 ratio: {lambda_2_avg:.3f}")
print(f"lambda_3 ratio: {lambda_3_avg:.3f}")