import torch 
from basenji2 import Basenji2
from deepsea import DeepSEA
from dilated import Dilated
from enformer import Enformer


if __name__ == "__main__":
    x = torch.randn(1, 4, 196608)
    basenji_input = torch.rand(1, 4, 131072)
    deepsea_input = torch.rand(1, 4, 1000)
    
    basenji2 = Basenji2()
    deepsea = DeepSEA()
    dilated = Dilated(out_channels=1536)
    enformer = Enformer()
    
    bas_out = basenji2(basenji_input)
    deep_sea_out = deepsea(deepsea_input)
    di_out = dilated(x)
    enf_out = enformer(x)
    
    print(bas_out.shape)
    print(deep_sea_out.shape)
    print(di_out.shape)
    print(enf_out.shape)
