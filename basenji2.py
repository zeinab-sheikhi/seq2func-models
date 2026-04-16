import torch 
import torch.nn as nn 

from modules import ConvBlock, DilatedConvs, PointWise, OutputHead
from utils import get_out_channels


class Basenji2(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 768, 
        n_towers: int = 6,
        n_layers: int = 11, 
        channel_ratio: float = 0.375, 
    ):
        super().__init__()
        ci = int(out_channels * channel_ratio)
        self.stem = nn.Sequential(
            ConvBlock(in_channels, ci, kernel_size=15), 
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.conv_tower = nn.ModuleList()
        channels = get_out_channels(start=ci, end=out_channels, n_blocks=n_towers + 1)
        for i in range(n_towers):
            self.conv_tower.append(
                nn.Sequential(
                    ConvBlock(channels[i], channels[i + 1], kernel_size=5), 
                    nn.MaxPool1d(kernel_size=2),
                )
            )
        
        self.dilated_conv = DilatedConvs(channels=out_channels, n_layers=n_layers, bottleneck=True)
        self.pointwise = PointWise(in_channels=out_channels)
        self.human_head = OutputHead(n_tracks=5313, in_channels=out_channels * 2)
        self.mouse_head = OutputHead(n_tracks=1643, in_channels=out_channels * 2)

    def forward(self, x: torch.Tensor, organism: str = "human"):
        x = self.stem(x)
        for block in self.conv_tower:
            x = block(x)
        x = self.dilated_conv(x)
        x = self.pointwise(x, trim=64)
        if organism == "human":
            return self.human_head(x)
        else: 
            return self.mouse_head(x)
