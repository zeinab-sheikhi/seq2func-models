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
        batch, seq_len, channels = x.shape
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



if __name__ == "__main__":
    pass