import torch
from torch import nn

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())

class PairUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=block(6,16);self.e2=block(16,32);self.mid=block(32,64)
        self.d2=block(96,32);self.d1=block(48,16);self.out=nn.Conv2d(16,5,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(nn.functional.max_pool2d(a,2));c=self.mid(nn.functional.max_pool2d(b,2))
        c=self.d2(torch.cat([nn.functional.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.out(self.d1(torch.cat([nn.functional.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))
