import torch 
from dilated import Dilated
from enformer import Enformer


if __name__ == "__main__":
    x = torch.randn(1, 4, 196608)   # (batch, channels, seq_len)
    enformer = Enformer()
    dilated = Dilated()
    # out = enformer(x)
    out = dilated(x)
    print(out.shape)
