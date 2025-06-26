import torch
import matplotlib.pyplot as plt

def display_pair(image, depth_map):   
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image.permute(1,2,0))
    ax1.axis("off")
    ax2.imshow(depth_map.permute(1,2,0))
    ax2.axis("off")
    plt.show()
    
def save_pair(image, depth_map, filepath):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image.permute(1,2,0))
    ax1.axis("off")
    ax2.imshow(depth_map.permute(1,2,0))
    ax2.axis("off")
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    
def color_transformer(tensor, cmap='inferno'):
    tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # normalize to [0,1]
    arr_colored = plt.get_cmap(cmap)(arr)[:, :, :, :3]  # drop alpha
    arr_colored = torch.from_numpy(arr_colored).float().cuda()
    arr_colored = arr_colored.permute(0, 3, 1, 2)
    return arr_colored