import torch 
from basenji2 import Basenji2
from dilated import Dilated
from enformer import Enformer


if __name__ == "__main__":
    x = torch.randn(1, 4, 196608)
    basenji_input = torch.rand(1, 4, 131072)
    
    basenji2 = Basenji2()
    dilated = Dilated(out_channels=1536)
    enformer = Enformer()
    
    bas_out = basenji2(basenji_input)
    enf_out = enformer(x)
    di_out = dilated(x)
    
    print(bas_out.shape)
    print(di_out.shape)
    print(enf_out.shape)
