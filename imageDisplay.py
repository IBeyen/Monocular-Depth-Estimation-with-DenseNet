import torch
import matplotlib.pyplot as plt

def display_pair(image, depth_map):   
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image.permute(1,2,0))
    ax1.axis("off")
    ax2.imshow(depth_map.permute(1,2,0))
    ax2.axis("off")
    plt.show()