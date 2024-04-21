import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import StepLR

def weighted_frobenious(W, AB, w=None):
    """
    Returns weighted frobenious distance between W and AB where "w" are the weights.
    I.e. returns projection loss.
    """
    if w is None:
        return torch.norm(W-AB)
    if len(w.shape) == 1:
        w = torch.ones_like(W)*w
    return torch.norm(torch.sqrt(w)*(W-AB))

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

def weighted_svd_inverse_weight(M, W):
    """
    Uses inverse-weight technique to come up with weighted SVD.
    M and W should be the same shape, M is the matrix, W are the weights
    """
    should_transpose = False
    if M.shape[0] > M.shape[1]:
        should_transpose = True
        M = M.T
        W = W.T
    W = torch.sqrt(W)
    M_right_inv = torch.linalg.solve(M.T @ M, M.T)
    # solves: W'M = W(*)M => W' = (W(*)M)M^-1
    W_prime = (W*M) @ M_right_inv
    U, S, V = get_svd(W*M)
    U = torch.linalg.solve(W_prime, U)
    if should_transpose:
        U, V = V, U
    return U, S, V

class WeightedSVD(nn.Module, object):
    """
    Computes weighted SVD given arbitrary weights using GD.
    Send to CUDA for maximum efficiency:
    w_svd = WeightedSVD(...).cuda()
    """
    def __init__(self, M, W, rank):
        super().__init__()
        out, inp = M.shape
        self.M = M.detach()
        self.W = W.detach()
        self.U = nn.Parameter(torch.randn(out, rank))
        self.V = nn.Parameter(torch.randn(inp, rank))
        self.S = nn.Parameter(torch.randn(rank))
        self.rank=rank
    
    def forward(self, x):
        return self.AB()@x

    def fit(self, iters=20000, lr=0.01, lambd=1):
        """ Fits WeightedSVD and returns U, S, V
        """
        errs = []
        optimizer = optim.Adam(self.parameters(), lr=lr)
        lr_scheduler = StepLR(optimizer, iters//10, gamma=0.8) # basically lr becomes 1/10 of start lr by end of training
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = weighted_frobenious(self.M, (self.U*self.S)@self.V.T, w=self.W)
            
            errs.append(loss.detach())
            
            ortho_penalty_u = weighted_frobenious(self.U.T@self.U, torch.eye(self.U.shape[1]))
            ortho_penalty_v = weighted_frobenious(self.V.T@self.V, torch.eye(self.V.shape[1]))
            
            loss = loss + lambd * (ortho_penalty_u + ortho_penalty_v)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        self.fix_USV()
        # return errs
        return self.U.detach().clone(), self.S.detach().clone(), self.V.detach().clone()
    
    def fix_USV(self):
        # Makes sure USV matches desired format
        self.U.data = self.U*torch.sign(self.S)
        self.S.data = torch.abs(self.S)
        perm = torch.argsort(self.S, descending=True)
        self.S.data = self.S[perm]
        self.U.data = self.U[:, perm]
        self.V.data = self.V[:, perm]