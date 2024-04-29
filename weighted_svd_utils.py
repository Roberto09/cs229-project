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

def get_svd(W, rank=None, use_cuda=False):
    """
    Returns SVD of W. If use_cuda, will force using cuda, which will be faster
    for big matrices.
    """
    rank = min(W.shape) if rank is None else rank
    # W = W.detach().clone()
    if use_cuda: W = W.cuda()
    svd = torch.svd(W)
    return svd.U[:, :rank], svd.S[:rank], svd.V[:, :rank]

def fit_to_svd_rowspace(V, M):
    """
    V is the orthogonal matrix from U S V^T.
    M is a matrix with the set of n column vectors we want to approximate using V as a basis.
    returns a matrix R with n column vectors, where the ith column vector is:
        R_i = argmin_{r} || M_i - V@r ||_F
    """
    R = torch.linalg.solve(V.T @ V, V.T) @ M
    return R
    
def rank_r(diag, r):
    new_diag = diag.clone()
    new_diag[r:] = 0
    return torch.diag(new_diag)

def get_svd_lora(M, r, use_cuda=False):
    """
    Returns SVD of rank r
    """
    u, s, v = get_svd(M, use_cuda=use_cuda)
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


def toy_seven_subspace_matrix(inp=80, out=100, r=6, noise=False):
    """
    Returns a toy matrix constructed by taking half of it's rows to live in an
    r-dimensional subpace and the other half to be in another r2-dimensional subspace.
    """
    assert out%7 == 0
    m1 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m2 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m3 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m4 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m5 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m6 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m7 = torch.randn(out//7, r) @ torch.randn(r, inp)
    m = torch.concat((m1, m2, m3, m4, m5, m6, m7))
    if noise:
        m += torch.randn(m.shape)*0.1
    #same_subspace_indices = [i for i in range(out//8)]
    return m#, same_subspace_indices

def toy_seven_subspace_matrix_unqual_size(inp=250, out=896, r=4, noise=True):
    """
    Returns a toy matrix constructed by taking half of it's rows to live in an
    r-dimensional subpace and the other half to be in another r2-dimensional subspace.
    """
    assert out%(16) == 0
    unit = out // (16)
    m1 = torch.randn(unit, r) @ torch.randn(r, inp)
    m2 = torch.randn(unit, r) @ torch.randn(r, inp)
    m3 = torch.randn(2*unit, r*2) @ torch.randn(r*2, inp)
    m4 = torch.randn(2*unit, r*2) @ torch.randn(r*2, inp)
    m5 = torch.randn(3*unit, r*3) @ torch.randn(r*3, inp)
    m6 = torch.randn(3*unit, r*3) @ torch.randn(r*3, inp)
    m7 = torch.randn(4*unit, r*4) @ torch.randn(r*4, inp)
    m = torch.concat((m1, m2, m3, m4, m5, m6, m7))
    if noise:
        m += torch.randn(m.shape)*0.1
    #same_subspace_indices = [i for i in range(out//8)]
    return m#, same_subspace_indices


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
    M_left_inv = torch.linalg.pinv(M)
    # solves: MW' = W(*)M => W' = M^-1(W(*)M)
    W_prime = M_left_inv @ (W*M)
    U, S, V = get_svd(W*M)
    # U = torch.linalg.solve(W_prime, U)
    V = (V.T @ torch.linalg.pinv(W_prime)).T 
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

    def fit(self, iters=2400, lr=0.01, lambd=1):
        """ Fits WeightedSVD and returns U, S, V
        """
        errs = []
        optimizer = optim.Adam(self.parameters(), lr=lr)
        lr_scheduler = StepLR(optimizer, iters//12, gamma=0.6) # basically lr becomes 1/10 of start lr by end of training
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            loss = weighted_frobenious(self.M, (self.U*self.S)@self.V.T, w=self.W)
            
            errs.append(loss.detach().clone())
            
            ortho_penalty_u = weighted_frobenious(self.U.T@self.U, torch.eye(self.U.shape[1]))
            ortho_penalty_v = weighted_frobenious(self.V.T@self.V, torch.eye(self.V.shape[1]))
            
            loss = loss + lambd * (ortho_penalty_u + ortho_penalty_v)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        self.fix_USV()
        return self.U.detach().clone(), self.S.detach().clone(), self.V.detach().clone()
    
    def fix_USV(self):
        # Makes sure USV matches desired format
        self.U.data = self.U*torch.sign(self.S)
        self.S.data = torch.abs(self.S)
        perm = torch.argsort(self.S, descending=True)
        self.S.data = self.S[perm]
        self.U.data = self.U[:, perm]
        self.V.data = self.V[:, perm]
