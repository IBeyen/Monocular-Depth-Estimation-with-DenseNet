import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = densenet169(weights="DEFAULT")
        
        """ Begin initialization of decoder """
        self.decoder_depth4 = nn.Sequential(nn.Conv2d(1920, 832, 3, padding="same"), 
                                             nn.Conv2d(832, 832, 3, padding="same")) 
        self.decoder_depth3 = nn.Sequential(nn.Conv2d(960, 416, 3, padding="same"), 
                                             nn.Conv2d(416, 416, 3, padding="same")) 
        self.decoder_depth2 = nn.Sequential(nn.Conv2d(480, 208, 3, padding="same"), 
                                             nn.Conv2d(208, 208, 3, padding="same")) 
        self.decoder_depth1 = nn.Sequential(nn.Conv2d(272, 104, 3, padding="same"), 
                                             nn.Conv2d(104, 104, 3, padding="same"))
        self.final_conv = nn.Conv2d(104, 1, 3, padding="same") 
        
    def forward(self, x):
        """ 
            We first run the input through the encoder 
            The layers of DenseNet go as follows 
            ["conv0", "norm0", "relu0", "pool0", "denseblock1", "transition1", "denseblock2", "transition2", "denseblock3", "transition3", "denseblock4", "norm5"]
            The ones we will be utilizing for skips are relu0, pool0, transition1, transition2 as defined by the paper
        """
        x = self.encoder.features.conv0(x)
        x = self.encoder.features.norm0(x)
        x = self.encoder.features.relu0(x)
        skip1 = x.clone()
        x = self.encoder.features.pool0(x)
        skip2 = x.clone()
        
        x = self.encoder.features.denseblock1(x)
        x = self.encoder.features.transition1(x)
        skip3 = x.clone()
        
        x = self.encoder.features.denseblock2(x)
        x = self.encoder.features.transition2(x)
        skip4 = x.clone()
    
        x = self.encoder.features.denseblock3(x)
        x = self.encoder.features.transition3(x)
        
        x = self.encoder.features.denseblock4(x)
        x = self.encoder.features.norm5(x)
        
        """" This is the end of the encoder and beginning of the decoder """
        x = F.interpolate(x, scale_factor=2, mode="bilinear") #Upsampling 1
        x = torch.cat([x, skip4], dim=1)
        x = self.decoder_depth4(x)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear") #Upsampling 2
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder_depth3(x)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear") #Upsampling 3
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder_depth2(x)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear") #Upsampling 4
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder_depth1(x)
        
        x = self.final_conv(x)
        
        return x