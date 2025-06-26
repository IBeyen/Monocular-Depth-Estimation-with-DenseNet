import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, crop, to_tensor
import numpy as np
from PIL import Image
import re

""" Takes in PIL image or Tensor and it's corresponding depth map and horizontally flips them with probability p. """
class FlipImageAndMap:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, depth_map):
        if np.random.rand() <= self.p:
            new_image = hflip(image)
            new_depth_map = hflip(depth_map)
            return new_image, new_depth_map
        return image, depth_map
       
""" Shuffles the RGB values with probability p. """ 
class ShuffleColors:
    def __init__(self, p=0.25):
        self.p = p
        
    def __call__(self, image, depth_map):
        if np.random.rand() <= self.p:
            """" The list of permutation below holds all permutations of the color channels except the default permutation """
            permutation = [torch.tensor([0, 2, 1]), torch.tensor([1, 0, 2]), torch.tensor([1, 2, 0]), torch.tensor([2, 0, 1]), torch.tensor([2, 1, 0])][np.random.randint(5)]
            return image[permutation, :, :], depth_map
        return image, depth_map

class RandomCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, depth_map):
        height, width = image.shape[1], image.shape[2]
        top = np.random.randint(0, height-self.size[0])
        left = np.random.randint(0, width-self.size[1])
        cropped_image = crop(image, top, left, self.size[0], self.size[1])
        cropped_depth_map = crop(depth_map, top, left, self.size[0], self.size[1])
        return cropped_image, cropped_depth_map
 
class ScaleValues:
    def __init__(self, image_scale=1, depth_scale=1):
        self.image_scale = image_scale
        self.depth_scale = depth_scale
        
    def __call__(self, image, depth_map):
        return image*self.image_scale, depth_map*self.depth_scale

class DepthClip:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def __call__(self, image, depth_map):
        new_depth_map = torch.clone(depth_map)
        new_depth_map[new_depth_map > self.upper] = self.upper
        new_depth_map[new_depth_map < self.lower] = self.lower
        return image, new_depth_map
        
"""" A custom version of torch's Compose that takes two inputs """
class Compose:
    def __init__(self, transformations):
        self.transformations = transformations
    
    def __call__(self, image, depth_map):
        for transformation in self.transformations:
            image, depth_map = transformation(image, depth_map)
        
        return image, depth_map

class NYUDataSet(Dataset):
    def __init__(self, data_paths, data_dir, transform=None):
        self.data_paths = pd.read_csv(data_paths, header=None)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data_paths.iloc[idx, 0])
        depth_map_path = os.path.join(self.data_dir, self.data_paths.iloc[idx, 1])
        image = to_tensor(Image.open(img_path))
        depth_map = to_tensor(Image.open(depth_map_path))
        if self.transform:
            image, depth_map = self.transform(image, depth_map)
        # if image.max() > 1:
        #     image /= 255
        # if depth_map.max() > 1:
        #     depth_map /= depth_map.max()
        return image.cuda(), depth_map.cuda()
    
    def get_original(self, idx):
        img_path = os.path.join(self.data_dir, self.data_paths.iloc[idx, 0])
        depth_map_path = os.path.join(self.data_dir, self.data_paths.iloc[idx, 1])
        image = to_tensor(Image.open(img_path))
        depth_map = to_tensor(Image.open(depth_map_path))
        return image.cuda(), depth_map.cuda()
    
    def get_name(self, idx):
        name = None
        string = self.data_paths.iloc[idx, 0]
        found = re.search("data\/nyu2_.*/([a-z].*)/([0-9]*).jpg", string)
        if found:
            name = found.group(1) + "_" + found.group(2)
        return name
