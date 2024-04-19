import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import pandas as pd

def weighted_frobenious(W, AB, w=None):
    """
    Returns weighted frobenious distance between W and AB where "w" are the weights.
    I.e. returns projection loss.
    """
    if w is None:
        return torch.norm(W-AB)
    assert len(w.shape) == 1
    return torch.norm(torch.sqrt(w.view(-1, 1))*(W-AB))

def get_svd(W, use_cuda=False):
    """
    Returns SVD of W. If use_cuda, will force using cuda, which will be faster
    for big matrices.
    """
    if use_cuda:
        W = W.detach().clone().cuda()
    svd = torch.svd(W)
    return svd.U, svd.S, svd.V

def rank_r(diag, r):
    new_diag = diag.clone()
    new_diag[r:] = 0
    return torch.diag(new_diag)

def get_svd_lora(M, r):
    """
    Returns SVD of rank r
    """
    u, s, v = get_svd(M)
    return u@rank_r(s, r)@v.T

def toy_correlated_matrix(inp=80, out=100):
    """
    Returns a toy correlated matrix, where ~50% of the rows are correlated
    """
    W = torch.randn(out, inp)
    w1 = W[0]
    correlated_idxs = [0]
    for i in range(1, out):
        if torch.randn(1) > 0:
            correlated_idxs.append(i)
            W[i] = w1 + torch.randn(w1.shape)*0.1
    return W, correlated_idxs

def toy_two_subspace_matrix(inp=80, out=100, r=32, r2=None, noise=False):
    """
    Returns a toy matrix constructed by taking half of it's rows to live in an
    r-dimensional subpace and the other half to be in another r2-dimensional subspace.
    """
    assert out%2 == 0
    r2 = r if r2 is None else r2
    m1 = torch.randn(out//2, r) @ torch.randn(r, inp)
    m2 = torch.randn(out//2, r2) @ torch.randn(r2, inp)
    m = torch.concat((m1, m2))
    if noise:
        m += torch.randn(m.shape)*0.1
    same_subspace_indices = [i for i in range(out//2)]
    return m, same_subspace_indices