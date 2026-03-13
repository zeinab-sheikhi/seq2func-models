import torch 
import torch.nn as nn 
import torch.nn.functional as F

from utils import get_out_channels


class AttentionPooling(nn.Module):
    def __init__(self, channels: int, pool_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.w = nn.Parameter(torch.eye(channels) * 2) 
    
    def forward(self, x: torch.Tensor):    
        batch, channels, seq_len = x.shape
        n_windows = (seq_len - self.pool_size) // self.stride + 1
        windows = []
        for i in range(self.pool_size):
            positions = x[:, :, i::self.stride]
            positions = positions[:, :, :n_windows]
            windows.append(positions)
        
        window = torch.stack(windows, dim=3)
        scores = torch.einsum("bcwp,cd->bdwp", window, self.w)
        weights = F.softmax(scores, dim=3)
        output = (weights * window).sum(dim=3)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=0.1),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)


class RConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.r_block = ConvBlock(in_channels, out_channels, kernel_size, stride, dilation)
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor):
        return self.r_block(x) + self.projection(x)


class Stem(nn.Module):
    def __init__(self, in_channels: int, channels: int):
        super().__init__()
        out_channels = channels // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7), 
            RConvBlock(out_channels, out_channels, kernel_size=1, dilation=1), 
            AttentionPooling(out_channels, pool_size=2, stride=2),
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)


class ConvTower(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int = 6):
        super().__init__()
        channels = get_out_channels(in_channels, out_channels, n_blocks + 1)
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            self.blocks.append(
                    nn.Sequential(
                    ConvBlock(channels[i], channels[i + 1], kernel_size=5, dilation=1),
                    RConvBlock(channels[i + 1], channels[i + 1], kernel_size=1),
                    AttentionPooling(channels=channels[i + 1], pool_size=2, stride=2),
                )
            )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x


class MHA(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        key_dim: int = 64,
        val_dim: int = 192,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.key_dim = key_dim
        self.val_dim = val_dim
        self.num_head = num_heads
        self.scale = self.key_dim ** -0.5

        self.q_proj = nn.Linear(channels, num_heads * key_dim)
        self.k_proj = nn.Linear(channels, num_heads * key_dim)
        self.v_proj = nn.Linear(channels, num_heads * val_dim)

        self.proj_out = nn.Linear(num_heads * val_dim, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, channels = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_head, self.key_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_head, self.key_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_head, self.val_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        scores = torch.matmul(attn, V)

        out = scores.transpose(2, 1).contiguous().view(batch_size, seq_len, self.num_head * self.val_dim)
        out = self.proj_out(out)
        return out


class MHABlock(nn.Module):
    def __init__(self, channels: int, key_dim: int = 64, num_heads: int = 8, dropout: float = 0.4):
        super().__init__()
        val_dim = channels // num_heads
        self.block = nn.Sequential(
            nn.LayerNorm(channels), 
            MHA(channels, num_heads, key_dim, val_dim), 
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class FeedForward(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels), 
            nn.Conv1d(in_channels, 2 * in_channels, kernel_size=1), 
            nn.Dropout(dropout), 
            nn.ReLU(), 
            nn.Conv1d(2 * in_channels, in_channels, kernel_size=1), 
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor):
        return x + self.block(x)


class PointWise(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        pass


if __name__ == "__main__":
    x = torch.randn(1, 4, 196608)  
    stem = Stem(in_channels=4, channels=1536)
    tower = ConvTower(in_channels=768, out_channels=1536)
    x = stem(x)
    x = tower(x)
    print(x.shape)
    n_params = sum(p.numel() for p in stem.parameters()) + sum(p.numel() for p in tower.parameters())
    print(n_params)

    
