import torch 
import torch.nn as nn 
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, channels: int, pool_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.w = nn.Parameter(torch.eye(channels) * 2) 
    
    def forward(self, x):    
        _, seq_len, _ = x.shape
        n_windows = (seq_len - self.pool_size) // self.stride + 1
        windows = []
        for i in range(self.pool_size):
            positions = x[:, i::self.stride, :]
            positions = positions[:, :n_windows, :]
            windows.append(positions)
        
        window = torch.stack(windows, dim=2)
        scores = window @ self.w
        weights = F.softmax(scores, dim=2)
        output = (weights * window).sum(dim=2)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=0.1),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        )
    
    def forward(self, x):
        return self.block(x)


class RConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int):
        super().__init__()
        self.r_block = ConvBlock(in_channels, out_channels, kernel_size, stride, dilation)
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        return self.r_block(x) + self.projection(x)


if __name__ == "__main__":
    x = torch.randn(1, 12, 4)  # batch=1, seq_len=12, channels=4

    # original Enformer settings
    pool = AttentionPooling(channels=4, pool_size=2, stride=2)
    print(pool(x).shape)  # (1, 6, 4) — halved the sequence

    # larger window, same stride
    pool = AttentionPooling(channels=4, pool_size=4, stride=2)
    print(pool(x).shape)  # (1, 5, 4) — more context per window

    # no downsampling
    pool = AttentionPooling(channels=4, pool_size=2, stride=1)
    print(pool(x).shape)  # (1, 11, 4) — sequence length roughly preserved