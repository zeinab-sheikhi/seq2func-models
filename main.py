import torch
import torch.nn as nn 

from modules import RelativePositionalEncoding


class MHA(nn.Module):
    # what do we need for this multiplication, the input would have the size (B, channels, 1536) = (B, C, L)
    def __init__(
        self,
        channels: int, 
        key_dim: int, 
        val_dim: int, 
        num_heads: int, 
        dropout: float = 0.4
    ):
        super().__init__()
        self.channels = channels
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.num_heads = num_heads
        assert key_dim % num_heads == 0, "The key dimension must be divisible by the number of heads"
        self.dk_head_dim = key_dim // num_heads
        self.dv_head_dim = val_dim // num_heads

        self.Wq = nn.Linear(channels, key_dim, bias=False)
        self.Wk = nn.Linear(channels, key_dim, bias=False)
        self.Wv = nn.Linear(channels, val_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(val_dim, channels)

        self.rel_pos = RelativePositionalEncoding(key_dim=self.dk_head_dim, num_features=val_dim)
        self.u = nn.Parameter(torch.zeros(num_heads, key_dim))
        self.v = nn.Parameter(torch.zeros(num_heads, key_dim))
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, channels = x.shape

        assert channels == self.channels, "Mismatch input dimension"
        
        Q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.dk_head_dim).transpose(1, 2) # [B, H, L, dk]
        K = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.dk_head_dim).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.dv_head_dim).transpose(1, 2)
        r_ij = self.rel_pos(seq_len) # (2 * seq_len - 1, key_dim)
        # r_ij = r_ij.unsqueeze(0).unsqueeze(1) # (1, 1, 2L-1, dk)

        # term1: content to content
        term1 = torch.matmul(Q, K.transpose(-2, -1)) # [B, H, L, L]
        print(term1.shape)

        # term2: content to position 
        term2 = torch.matmul(Q, r_ij.transpose(-2, -1)) # [B, H, L, 2L-1]
        print(term2.shape)

        # term3: position agnostic key preference
        self.u = self.u.unsqueeze(0).unsqueeze(2) # [1, H, 1, dk]
        term3 = torch.matmuk(self.u, K.transpose(-2, -1)) # [B, H, 1, L]

        # term4: position agnostic distance preference
        self.v = self.v.unsqueeze(0).unsqueeze(2) # [1, H, 1, dk]
        term4 = torch.matmul(self.v, r_ij.transpose(-2, -1)) # [1, H, 1, 2L-1]

    def _rel_shift(self, x: torch.Tensor):
        batch, heads, seq_len, full_len = x.shape
        x = F.pad(x, (1, 0)) # [B, H, L, 2L]
        x = x.view(batch, heads, full_len +1, seq_len) # [B, H, 2L, L]
        x = x[:, :, 1:, :] # [B, H, 2L-1, L]
        x = x[:, :, :seq_len, :]
        return x # [B, H, L, L]


def main():
    x = torch.tensor(torch.rand(1, 10, 16))
    mha = MHA(channels=16, key_dim=8, val_dim=36, num_heads=2)
    print(mha(x))



if __name__ == "__main__":
    main()
