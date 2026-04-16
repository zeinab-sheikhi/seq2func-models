import torch
import torch.nn as nn 

from modules import (
    Stem, 
    ConvTower,  
    DilatedConvs, 
    PointWise, 
    OutputHead,
)


class Dilated(nn.Module):
    def __init__(
        self,
        in_channels: int = 4, 
        channels: int = 768, 
    ):
        super().__init__()
        self.stem = Stem(in_channels, channels, dilated=True)
    
    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        return x
