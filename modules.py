import torch 
import torch.nn as nn 
import torch.nn.functional as F

from utils import (
    get_out_channels,
    positional_features_central_mask,
    positional_features_exponential,
    positional_features_gamma,
    softplus,
)


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


class RelativePositionalEncoding(nn.Module):
    def __init__(self, key_dim: int, num_features: int = 192):
        super().__init__()
        self.key_dim = key_dim 
        self.num_features = num_features
        self.feat_size = num_features // 6
        self.W_R = nn.Linear(num_features, key_dim, bias=False)
    
    def _compute_basis(self, positions: torch.Tensor):
        exp = positional_features_exponential(positions, self.feat_size)
        mask = positional_features_central_mask(positions, self.feat_size)
        gamma = positional_features_gamma(positions, self.feat_size)
        return torch.cat([exp, mask, gamma], dim=1)
    
    def forward(self, seq_len: int):
        positions = torch.arange(
            -(seq_len - 1), seq_len,
            dtype=torch.float32, 
            device=self.W_R.weight.device,
        )
        basis = self._compute_basis(positions)
        return self.W_R(basis)  # (2 * seq_len - 1, key_dim)


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
        self.num_heads = num_heads
        self.scale = self.key_dim ** -0.5

        self.q_proj = nn.Linear(channels, num_heads * key_dim)
        self.k_proj = nn.Linear(channels, num_heads * key_dim)
        self.v_proj = nn.Linear(channels, num_heads * val_dim)

        self.proj_out = nn.Linear(num_heads * val_dim, channels)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoding = RelativePositionalEncoding(key_dim)
        self.u = nn.Parameter(torch.zeros(num_heads, key_dim)) 
        self.v = nn.Parameter(torch.zeros(num_heads, key_dim))
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, channels = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)  # [B, H, L, dk]
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)  # [B, H, L, dk]
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.val_dim).transpose(1, 2)  # [B, H, L, dv]

        R = self.pos_encoding(seq_len).to(x.device)  # [2 * L - 1, dk]

        # term1: content to content 
        term1 = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, L, L]

        # term2: content to position 
        term2 = torch.matmul(Q, R.transpose(-2, -1))  # [B, H, L, 2 * L -1]
        term2 = self._rel_shift(term2)  # [B, H, L, L]

        # term3: position-agnostic key preference
        u = self.u.unsqueeze(0).unsqueeze(2)  # [1, H, 1, dk]
        term3 = torch.matmul(u, K.transpose(-2, -1))  # [B, H, 1, L]

        # term4: position-agnostic distance preference
        v = self.v.unsqueeze(0).unsqueeze(2)
        term4 = torch.matmul(v, R.transpose(-2, -1)) # [1, H, 1, 2 * L - 1]
        term4 = self._rel_shift(term4)

        attn = (term1 + term2 + term3 + term4) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.val_dim)
        return self.proj_out(out)

    def _rel_shift(self, x):
        batch, heads, seq_len, full_len = x.shape
        x = F.pad(x, (1, 0)) # [B, H, L, 2L]
        x = x.view(batch, heads, full_len + 1, seq_len) # [B, H, 2L, L]
        x = x[:, :, 1:, :] # [B, H, 2L - 1, L]
        return x[:, :, :seq_len, :] # [B, H, L, L]


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
        self.norm = nn.LayerNorm(in_channels)
        self.block = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2), 
            nn.Dropout(dropout), 
            nn.ReLU(), 
            nn.Linear(2 * in_channels, in_channels), 
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor):
        return x + self.block(self.norm(x))


class Transformer(nn.Module):
    def __init__(self, num_layers: int, channels: int, key_dim: int = 64, num_heads: int = 8, dropout: float = 0.4):
        super().__init__()
        self.mha_blocks = nn.ModuleList([
            MHABlock(
                channels=channels,
                key_dim=key_dim, 
                num_heads=num_heads, 
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
        self.ff_blocks = nn.ModuleList([
            FeedForward(in_channels=channels, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        for mha, ff in zip(self.mha_blocks, self.ff_blocks):
            x = mha(x)
            x = ff(x)
        return x


class PointWise(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.05, ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1),
            nn.Dropout(dropout), 
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor):
        # trim 320 on both edges
        x = x[:, 320 : -320, :]
        x = x.permute(0, 2, 1)
        x = self.block(x)
        return x


class OutputHead(nn.Module):
    def __init__(self, n_tracks: int, in_channels: int = 3072):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, n_tracks, kernel_size=1)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.softplus(x)
        return x


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


if __name__ == "__main__":
    x = torch.randn(1, 4, 196608)   # (batch, channels, seq_len)
    enformer = Enformer()
    out = enformer(x)
    print(out.shape)
