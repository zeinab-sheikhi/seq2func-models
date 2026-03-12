import torch 
import torch.nn as nn 
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.w = nn.Parameter(torch.eye(channels) * 2) 
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        
        x1 = x[:, 0::2, :]  # (batch, seq_len // 2, channels)
        x2 = x[:, 1::2, :]  # (batch, seq_len // 2, channels)
        window = torch.stack([x1, x2], dim=2)

        scores = window @ self.w
        weights = F.softmax(scores, dim=2)
        output = (weights * window).sum(dim=2)
        return output

