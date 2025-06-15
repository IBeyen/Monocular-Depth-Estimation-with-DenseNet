import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pytorch_msssim import ssim
from loss import Loss
from dataSetup import NYUDataSet, Compose, ShuffleColors, FlipImageAndMap
from model import Model
import time
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 2
EPOCH_START = 0

transformations = Compose([ShuffleColors(), FlipImageAndMap()])
train_dataset = NYUDataSet("nyu_data\\data\\nyu2_train.csv", "nyu_data", transformations)
val_dataset = NYUDataSet("nyu_data\\data\\nyu2_val.csv", "nyu_data", transformations)
test_dataset = NYUDataSet("nyu_data\\data\\nyu2_test.csv", "nyu_data")

model = Model().cuda()
loss_fn = Loss(1)
optimizer = torch.optim.Adam(model.parameters(), 0.000075)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset)
test_loader = DataLoader(test_dataset)

model_dir = "Test_6_15_2025_1"

if os.path.isdir(model_dir):
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pth")))
    optimizer.load_state_dict(torch.load(os.path.join(model_dir, "optimizer_state.pth")))

writer = SummaryWriter(log_dir=model_dir+"/logs")

def color_transformer(tensor, cmap='inferno'):
    tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # normalize to [0,1]
    arr_colored = plt.get_cmap(cmap)(arr)[:, :, :, :3]  # drop alpha
    arr_colored = torch.from_numpy(arr_colored).float().cuda()
    arr_colored = arr_colored.permute(0, 3, 1, 2)
    return arr_colored

""" Get images to save for tracking progress """
img_0, depth_map_0 = val_dataset.get_original(0)
img_0 = img_0.unsqueeze(0)
depth_map_0 = depth_map_0.unsqueeze(0)
depth_map_0 = color_transformer(depth_map_0)

img_1, depth_map_1 = val_dataset.get_original(200)
img_1 = img_1.unsqueeze(0)
depth_map_1 = depth_map_1.unsqueeze(0)
depth_map_1 = color_transformer(depth_map_1)

img_2, depth_map_2 = val_dataset.get_original(410)
img_2 = img_2.unsqueeze(0)
depth_map_2 = depth_map_2.unsqueeze(0)
depth_map_2 = color_transformer(depth_map_2)

for epoch in range(EPOCH_START, 10):
    model.train()
    epoch_start = time.time()
    train_loss_sum = 0
    train_ssim_sum = 0
    iter_group_start = time.time() # This variable is used to keep track of how long it takes to do 100 iterations
    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.float()
        y = y.float()
        
        y_pred = model.forward(x)
        
        # y_true = 1000/y
        y_true = F.avg_pool2d(y, 2)
        
        optimizer.zero_grad()
        loss = loss_fn(y_true, y_pred)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
        optimizer.step()
        
        train_loss_sum += loss.detach()
        train_ssim_sum += ssim(y_true, y_pred, size_average=True, data_range=1).detach()
            
        del x
        del y
        del y_true
        del y_pred
        
        """ Begin logging section """
        
        if i % 250 == 0 and i != 0:
            writer.add_scalar("Training Loss over iteration", train_loss_sum/250, epoch*len(train_loader) + i)
            writer.add_scalar("Training SSIM", train_ssim_sum/250, epoch*len(train_loader) + i)
            train_loss_sum = 0
            train_ssim_sum = 0
            """ Save time """
            writer.add_scalar("Time it took to run last 250 iterations (s)", time.time()-iter_group_start, epoch*len(train_loader) + i)
            iter_group_start = time.time()
            writer.flush()
        if i % 2500 == 0:
            model.eval()
            val_loss_sum = 0
            val_ssim_sum = 0
            for val_batch in val_loader:
                x, y = val_batch
                x = x.float()
                y = y.float()
                
                y_pred = model.forward(x)
                y_pred = F.interpolate(y_pred, scale_factor=2)
                
                # y_true = 1000/y
                
                loss = loss_fn(y, y_pred)
                val_loss_sum += loss.detach()
                
                val_ssim_sum += ssim(y, y_pred, size_average=True, data_range=1).detach()
                
                del x
                del y
                # del y_true
                del y_pred
                
            val_loss_avg = val_loss_sum / len(val_dataset)
            writer.add_scalar("Validation Loss over iteration", val_loss_avg, epoch*len(train_loader) + i)
            val_ssim_avg = val_ssim_sum / len(val_dataset)
            writer.add_scalar("Validation SSIM", val_ssim_avg, epoch*len(train_loader) + i)
            
            """ Here we do the original image, depth map, and predicted depth map side by side"""         
            prediction = F.interpolate(model(img_0), scale_factor=2)
            prediction = color_transformer(prediction)
            grid_0 = torchvision.utils.make_grid(torch.concatenate([img_0, depth_map_0, prediction]))
            writer.add_image("Image1, Map1, Predicted1", grid_0, epoch*len(train_loader) + i)
            
            prediction = F.interpolate(model(img_1), scale_factor=2)
            prediction = color_transformer(prediction)
            grid_1 = torchvision.utils.make_grid(torch.concatenate([img_1, depth_map_1, prediction]))
            writer.add_image("Image2, Map2, Predicted2", grid_1, epoch*len(train_loader) + i)
            
            prediction = F.interpolate(model(img_2), scale_factor=2)
            prediction = color_transformer(prediction)
            grid_2 = torchvision.utils.make_grid(torch.concatenate([img_2, depth_map_2, prediction]))
            writer.add_image("Image3, Map3, Predicted3", grid_2, epoch*len(train_loader) + i)
            
            writer.flush()
            
            torch.save(model.state_dict(), model_dir + "/model_weights.pth")
            torch.save(optimizer.state_dict(), model_dir + "/optimizer_state.pth")
            
            model.train()
            print(f"{i/len(train_loader)*100:.2f}% through epoch {epoch+1}")
            torch.cuda.empty_cache()
            
            """ End logging section """
        
    epoch_end = time.time()
    writer.add_scalar("Time taken for epoch", (epoch_end-epoch_start)/60**2, epoch+1)
    writer.flush()
    
model.eval()
test_loss_sum = 0
test_ssim_sum = 0
for test_batch in test_loader:
    x, y = test_batch
    x = x.float()
    y = y.float()
    
    y_pred = model.forward(x)
    y_pred = F.interpolate(y_pred, scale_factor=2)
    
    # y_true = 1000/y
    
    loss = loss_fn(y, 10000*y_pred)
    val_loss_sum += loss.detach()
    
    val_ssim_sum += ssim(y, 10000*y_pred, size_average=True, data_range=1).detach()
    
    del x
    del y
    # del y_true
    del y_pred
    
test_loss_avg = val_loss_sum / len(test_dataset)
writer.add_scalar("Test Loss", test_loss_avg)
test_ssim_avg = val_ssim_sum / len(test_dataset)
writer.add_scalar("Test SSIM", test_ssim_avg)
    
writer.close()