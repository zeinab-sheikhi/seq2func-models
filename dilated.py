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
        out_channels: int = 768, 
        n_layers: int = 11, 
        dilation_rate: int = 1, 
        dilation_factor: float = 1.5, 
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = Stem(in_channels, out_channels, dilated=True)
        self.conv_tower = ConvTower(out_channels // 2, out_channels, dilated=True)
        self.dilated_convs = DilatedConvs(out_channels, n_layers, dilation_rate, dilation_factor, dropout)
        self.pointwise = PointWise(out_channels)
        self.human_head = OutputHead(n_tracks=5313) 
        self.mouse_head = OutputHead(n_tracks=1643)
    
    def forward(self, x: torch.Tensor, organism: str = "human"):
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.dilated_convs(x)
        x = self.pointwise(x)
        if organism == "human":
            return self.human_head(x)
        else:
            return self.mouse_head(x)
