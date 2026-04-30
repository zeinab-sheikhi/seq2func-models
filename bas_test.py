import torch
import torch.nn as nn 


class ConvBlock(nn.Module):
    def __init__(
        self, 
        conv_kernel_size: int = 11, 
        in_channels: int = 4,
        filter_sizes: list[int] | None = None, 
        pooling_kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        filter_sizes = filter_sizes or [128, 128, 192, 256]
        pooling_kernel_sizes = pooling_kernel_sizes or [2, 4, 4, 4]
        
        blocks = []
        current_channel = in_channels
        for out_channel, pool_size in zip(filter_sizes, pooling_kernel_sizes):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(current_channel, out_channel, kernel_size=conv_kernel_size, padding="same"), 
                    nn.BatchNorm1d(num_features=out_channel),
                    nn.ReLU(), 
                    nn.MaxPool1d(kernel_size=pool_size),
                ),
            )
            current_channel = out_channel

        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class DilatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, conv_kernel: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, conv_kernel, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)


class DilatedConvBlock(nn.Module):
    def __init__(
        self, 
        dilation_base: int = 2, 
        n_layers: int = 7, 
        in_channels: int = 256, 
        out_channels: int = 128,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1, n_layers + 1):
            current_channel = in_channels + out_channels * (i - 1)
            self.layers.append(
                DilatedConv(
                    in_channels=current_channel, 
                    out_channels=out_channels, 
                    dilation=dilation_base ** (i - 1), 
                )
            )
    
    def forward(self, x: torch.Tensor):
        outputs = [x]
        for block in self.layers:
            input_x = torch.cat(outputs, dim=1)
            x = block(input_x)
            outputs.append(x)
        return torch.cat(outputs, dim=1)




if __name__ == "__main__":
    # x = torch.rand(1, 4, 131072)

    # number of channels increases, but the length stays the same
    x = torch.rand(1, 256, 1024)
    
    # layer1 = DilatedConv(in_channels=256, out_channels=128, dilation=1, conv_kernel=3)
    # out1 = layer1([x]) # [1, 128, 1024]
    # layer2 = DilatedConv(in_channels=128 * 3, out_channels=128, dilation=2, conv_kernel=3)
    # out2 = layer2([x, out1]) # [1, 128, 1024]
    # layer3 = DilatedConv(in_channels=128 * 4, out_channels=128, dilation=4, conv_kernel=3)
    # out3 = layer3([x, out1, out2]) # [1, 128, 1024]
    # layer4 = DilatedConv(in_channels=128 * 5, out_channels=128, dilation=8, conv_kernel=3)
    # out4 = layer4([x, out1, out2, out3]) # [1, 128, 1024]
    # layer5 = DilatedConv(in_channels=128 * 6, out_channels=128, dilation=16)
    # out5 = layer5([x, out1, out2, out3, out4]) # [1, 128, 1024]
    # layer6 = DilatedConv(in_channels=128 * 7, out_channels=128, dilation=32)
    # out6 = layer6([x, out1, out2, out3, out4, out5]) # [1, 128, 1024]
    # layer7 = DilatedConv(in_channels=128 * 8, out_channels=128, dilation=64)
    # out7 = layer7([x, out1, out2, out3, out4, out5, out6]) # [1, 128, 1024]

    conv_block = DilatedConvBlock()
    out = conv_block(x)
    # print(out.shape)