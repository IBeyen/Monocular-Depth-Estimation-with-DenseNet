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

transformations = Compose([ShuffleColors(), FlipImageAndMap()])
train_dataset = NYUDataSet("nyu_data\\data\\nyu2_train.csv", "nyu_data", transformations)
val_dataset = NYUDataSet("nyu_data\\data\\nyu2_val.csv", "nyu_data", transformations)
test_dataset = NYUDataSet("nyu_data\\data\\nyu2_test.csv", "nyu_data")

model = Model().cuda()
loss_fn = Loss(0.1)
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

train_loader = DataLoader(train_dataset, shuffle=True)
val_loader = DataLoader(val_dataset, shuffle=True)
test_loader = DataLoader(test_dataset)

model_dir = "Test4_6_2_2025"

writer = SummaryWriter(log_dir=model_dir+"/logs")

color_transformer = torchvision.transforms.Lambda(lambda x: x.repeat(1,3,1,1))
            
img, depth_map = val_dataset[0]
img = img.unsqueeze(0)
depth_map = depth_map.unsqueeze(0)
prediction = F.interpolate(model(img), scale_factor=2)
depth_map = color_transformer(depth_map)
prediction = color_transformer(prediction)
grid_0 = torchvision.utils.make_grid(torch.concatenate([img, depth_map, prediction]))
writer.add_image("Image1, Map1, Predicted1", grid_0, 0)

img, depth_map = val_dataset[200]
img = img.unsqueeze(0)
depth_map = depth_map.unsqueeze(0)
prediction = F.interpolate(model(img), scale_factor=2)
depth_map = color_transformer(depth_map)
prediction = color_transformer(prediction)
grid_1 = torchvision.utils.make_grid(torch.concatenate([img, depth_map, prediction]))
writer.add_image("Image2, Map2, Predicted2", grid_1, 0)

img, depth_map = val_dataset[410]
img = img.unsqueeze(0)
depth_map = depth_map.unsqueeze(0)
prediction = F.interpolate(model(img), scale_factor=2)
depth_map = color_transformer(depth_map)
prediction = color_transformer(prediction)
grid_2 = torchvision.utils.make_grid(torch.concatenate([img, depth_map, prediction]))
writer.add_image("Image3, Map3, Predicted3", grid_2, 0)

writer.flush()

for epoch in range(1):
    model.train()
    epoch_start = time.time()
    train_loss_sum = 0
    train_ssim_sum = 0
    iter_group_start = time.time() # This variable is used to keep track of how long it takes to do 100 iterations
    for i, batch in enumerate(train_loader):
        
        """ Begin logging section """
        
        if i % 100 == 0 and i != 0:
            writer.add_scalar("Training Loss over iteration", train_loss_sum/100, epoch*len(train_loader) + i)
            writer.add_scalar("Training SSIM", train_ssim_sum/100, epoch*len(train_loader) + i)
            train_loss_sum = 0
            train_ssim_sum = 0
            """ Save time """
            writer.add_scalar("Time it took to run last 100 iterations (s)", time.time()-iter_group_start, epoch*len(train_loader) + i)
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
                
                y_true = 1000/y
                
                loss = loss_fn(y_true, y_pred)
                val_loss_sum += loss.detach()
                
                val_ssim_sum += ssim(y_true, y_pred, size_average=True).detach()
                
            val_loss_avg = val_loss_sum / len(val_dataset)
            writer.add_scalar("Validation Loss over iteration", val_loss_avg, epoch*len(train_loader) + i)
            val_ssim_avg = val_ssim_sum / len(val_dataset)
            writer.add_scalar("Validation SSIM", val_ssim_avg, epoch*len(train_loader) + i)
            
            """ Here we do the original image, depth map, and predicted depth map side by side"""
            color_transformer = torchvision.transforms.Lambda(lambda x: x.repeat(1,3,1,1))
            
            img, depth_map = val_dataset[0]
            img = img.unsqueeze(0)
            depth_map = depth_map.unsqueeze(0)
            prediction = F.interpolate(model(img), scale_factor=2)
            depth_map = color_transformer(depth_map)
            prediction = color_transformer(prediction)
            grid_0 = torchvision.utils.make_grid(torch.concatenate([img, depth_map, prediction]))
            writer.add_image("Image1, Map1, Predicted1", grid_0, epoch*len(train_loader) + i)
            
            img, depth_map = val_dataset[200]
            img = img.unsqueeze(0)
            depth_map = depth_map.unsqueeze(0)
            prediction = F.interpolate(model(img), scale_factor=2)
            depth_map = color_transformer(depth_map)
            prediction = color_transformer(prediction)
            grid_1 = torchvision.utils.make_grid(torch.concatenate([img, depth_map, prediction]))
            writer.add_image("Image2, Map2, Predicted2", grid_1, epoch*len(train_loader) + i)
            
            img, depth_map = val_dataset[410]
            img = img.unsqueeze(0)
            depth_map = depth_map.unsqueeze(0)
            prediction = F.interpolate(model(img), scale_factor=2)
            depth_map = color_transformer(depth_map)
            prediction = color_transformer(prediction)
            grid_2 = torchvision.utils.make_grid(torch.concatenate([img, depth_map, prediction]))
            writer.add_image("Image3, Map3, Predicted3", grid_2, epoch*len(train_loader) + i)
            
            writer.flush()
            
            torch.save(model.state_dict(), model_dir + "/model_weights.pth")
            torch.save(optimizer.state_dict(), model_dir + "/optimizer_state.pth")
            
            model.train()
            print(f"{i/len(train_loader)*100:.2f}% through epoch {epoch+1}")
            
            """ End logging section """
            
            
        """ Begin training code """
        
        x, y = batch
        x = x.float()
        y = y.float()
        
        y_pred = model.forward(x)
        
        y_true = 1000/y
        y_true = F.avg_pool2d(y, 2)
        
        optimizer.zero_grad()
        loss = loss_fn(y_true, y_pred)
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.detach()
        train_ssim_sum += ssim(y_true, y_pred, size_average=True).detach()
        
    epoch_end = time.time()
    writer.add_scalar("Time taken for epoch", epoch_end-epoch_start, epoch+1)
    writer.flush()
writer.close()