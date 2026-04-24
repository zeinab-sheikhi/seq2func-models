import torch 
import torch.nn as nn 

from typing import List


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        pooling_kernel_size: int, 
        pooling_stride: int, 
        dropout: float, 
        pooling: bool = True, 
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride), 
            nn.ReLU(), 
            nn.MaxPool1d(pooling_kernel_size, pooling_stride) if pooling else nn.Identity(), 
            nn.Dropout(dropout), 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepSEA(nn.Module):
    def __init__(
        self,
        input_length: int = 1000, 
        channels: List[int] = [4, 320, 480, 960],
        n_ff_neurons: int = 925,
        n_out_tracks: int = 919, 
        conv_kernel_size: int = 8, 
        conv_stride: int = 1, 
        pooling_kernel_size: int = 4, 
        pooling_stride: int = 4, 
        dropouts: List[float] = [0.2, 0.2, 0.5],
    ):
        super().__init__()
        assert len(channels) - 1 == len(dropouts), "Number of dropouts must equal number of conv blocks!"
        
        self.channels = channels
        self.conv_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=channels[i], 
                    out_channels=channels[i + 1], 
                    kernel_size=conv_kernel_size, 
                    stride=conv_stride, 
                    pooling=i != len(channels) - 2, 
                    pooling_kernel_size=pooling_kernel_size, 
                    pooling_stride=pooling_stride,
                    dropout=dropouts[i],
                )
            )

        flat_size = self._get_flat_size(input_length)
        self.linear1 = nn.Linear(flat_size, n_ff_neurons)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(n_ff_neurons, n_out_tracks)
    
    def forward(self, x: torch.Tensor):
        for block in self.conv_blocks:
            x = block(x)
        B, C, L = x.size()
        x = x.view(B, C * L)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x 
    
    def _get_flat_size(self, input_length: int) -> int:
        dummy = torch.zeros(1, self.channels[0], input_length)
        for block in self.conv_blocks:
            dummy = block(dummy)
        return dummy.flatten(start_dim=1).shape[1]
