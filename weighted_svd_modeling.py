import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

class LowRankMatrix(nn.Module):
    def __init__(self, inp, out, rank):
        super().__init__()
        self.inp, self.out = inp, out
        self.A = nn.Parameter(torch.randn(out, rank))
        self.B = nn.Parameter(torch.randn(rank, inp))
        self.rank=rank

    def forward(self, x):
        return self.AB()@x

    def AB(self):
        return self.A @ self.B
    
class Mask(nn.Module):
    def __init__(self, out, init_mask=False):
        super().__init__()
        vals = torch.randn(out) if not init_mask else torch.zeros(out)
        self.v_mask = nn.Parameter(vals)
    
    def mask1(self):
        return torch.nn.functional.sigmoid(self.v_mask)
    
    def mask2(self):
        return 1 - self.mask1()
    
class LowRankSplit(nn.Module):
    def __init__(self, inp, out, rank, rank2=None, init_mask=False):
        super().__init__()
        self.inp, self.out = inp, out
        rank2 = rank if rank2 is None else rank2
        self.mat1 = LowRankMatrix(inp, out, rank)
        self.mat2 = LowRankMatrix(inp, out, rank2)
        self.mask = Mask(out, init_mask=init_mask)

    def forward(self):
        # Returns AB matrix approximation from low-rank split
        assert not self.training
        res = torch.empty(self.out, self.inp)
        m1 = self.mask.mask1() >= 0.5
        m2 = self.mask.mask2() > 0.5
        assert torch.all(torch.logical_xor(m1, m2))
        # import pdb; pdb.set_trace()
        res[m1] = self.mat1.AB()[m1]
        res[m2] = self.mat2.AB()[m2]
        return res
    
    def AB1(self):
        return self.mat1.AB()

    def mask1(self):
        return self.mask.mask1()
    
    def AB2(self):
        return self.mat2.AB()

    def mask2(self):
        return self.mask.mask2()

    def fit_svd(self, M, elems_override=None):
        elems_1 = self.mask1()>=0.5 if elems_override is None else elems_override
        elems_2 = self.mask2()>0.5 if elems_override is None else elems_override
        if elems_override is None:
            assert torch.all(torch.logical_xor(elems_1, elems_2))
        
        svd1 = torch.svd(M[elems_1])
        self.mat1.A.data[elems_1] = (svd1.U @ rank_r(svd1.S, self.mat1.rank))[:, :self.mat1.rank] 
        if elems_override is None: self.mat1.A.data[elems_2] = 0
        self.mat1.B.data = svd1.V.T[:self.mat1.rank]

        svd2 = torch.svd(M[elems_2])
        self.mat2.A.data[elems_2] = (svd2.U @ rank_r(svd2.S, self.mat2.rank))[:, :self.mat2.rank]
        if elems_override is None: self.mat2.A.data[elems_1] = 0
        self.mat2.B.data = svd2.V.T[:self.mat2.rank]

