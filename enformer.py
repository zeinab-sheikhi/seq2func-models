import torch
import torch.nn as nn

from modules import (
    Stem, 
    ConvTower, 
    Transformer, 
    PointWise,
    OutputHead,
)


class Enformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1536,
        n_conv_tower_blocks: int = 6, 
        n_transformer_layers: int = 11, 
        key_dim: int = 64, 
        num_heads: int = 8,
        trans_dropout: float = 0.1, 
    ):
        super().__init__()
        self.stem = Stem(in_channels, out_channels)
        self.conv_tower = ConvTower(out_channels // 2, out_channels, n_conv_tower_blocks)
        self.transformer = Transformer(n_transformer_layers, out_channels, key_dim, num_heads, trans_dropout)
        self.pointwise = PointWise(out_channels)
        self.human_head = OutputHead(n_tracks=5313)
        self.mouse_head = OutputHead(n_tracks=1643)
    
    def forward(self, x: torch.Tensor, organism: str = "human"):
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.transformer(x)
        x = self.pointwise(x)
        if organism == "human":
            return self.human_head(x)
        else: 
            return self.mouse_head(x)
