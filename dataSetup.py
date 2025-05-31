import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip
from torchvision.io import decode_image

""" Takes in PIL image or Tensor and it's corresponding depth map and horizontally flips them with probability p. """
def FlipImageAndMap(image, depth_map, p=0.5):
    if torch.rand() <= p:
        new_image = hflip(image)
        new_depth_map = hflip(depth_map)
        return new_image, new_depth_map
    return image, depth_map
       
""" Shuffles the RGB values with probability p. """ 
def ShuffleColors(image, depth_map, p=0.5):
    if torch.rand() <= p:
        permutation = [torch.tensor([0, 2, 1]), torch.tensor([1, 0, 2]), torch.tensor([1, 2, 0]), torch.tensor([2, 0, 1]), torch.tensor([2, 1, 0])][torch.randint(0, 6)]
        return image[permutation, :, :], depth_map
    return image, depth_map

class NYUDataSet(Dataset):
    def __init__(self, data_paths, data_dir, transform=None):
        self.data_paths = pd.read_csv(data_paths)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data_paths.iloc[idx, 0])
        depth_map_path = os.path.join(self.data_dir, self.data_paths.iloc[idx, 1])
        image = decode_image(img_path)
        depth_map = decode_image(depth_map_path)
        if self.transform:
            image, depth_map = self.transform(image, depth_map)
        return image, depth_map