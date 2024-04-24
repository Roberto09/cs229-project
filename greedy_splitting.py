import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Optional
import copy

from weighted_svd_utils import weighted_frobenious, get_svd, get_svd_lora

class SubMatrix():
    def __init__(self, orig_matrix, rows : Optional[List[int]] = None):
        if type(orig_matrix) == SubMatrix: orig_matrix = orig_matrix.get_orig_torch_matrix()
        self.orig_matrix = orig_matrix.detach().clone()
        if rows is None: rows = list(range(len(orig_matrix)))
        self.rows = copy.deepcopy(rows)

    @property
    def num_rows(self):
        return len(self.rows)

    @property
    def num_cols(self):
        assert self.has_rows()
        return self.orig_matrix.shape[1]

    def add_row(self, row:int):
        assert row not in self.rows
        self.rows.append(row)

    def remove_row(self, row:int):
        assert row in self.rows
        self.rows.remove(row)

    def get_rows(self):
        return [(row, self.orig_matrix[row]) for row in self.rows]

    def get_orig_torch_matrix(self):
        return self.orig_matrix.detach().clone()

    def get_dense_torch_matrix(self):
        return self.orig_matrix[self.rows].detach().clone()

    def has_rows(self):
        return self.num_rows != 0
    
    def sort_rows(self):
        self.rows.sort()

    def get_flops_full_matrix(self):
        return 2 * self.num_rows * self.num_cols
    
    def copy(self):
        return copy.deepcopy(self)

def flops_from_shape(inp, out, rank):
    return 2 * rank * (inp + out)

def flops_from_svd(U, S, V):
    if U is None or S is None or V is None: return 0
    assert U.shape[1] == S.shape[0] == V.shape[1], "SVD had unexpected shape"
    ret = flops_from_shape(inp=V.shape[0], out=U.shape[0], rank=S.shape[0])
    return ret

# TODO: More accurate flop calculation everywhere

def find_optimal_rank_allocation_F2(mat1, mat2, flops):
    mat1_flops = 2 * mat1.shape[0] * mat1.shape[1] if mat1 != None else 0.0
    mat2_flops = 2 * mat2.shape[0] * mat2.shape[1] if mat2 != None else 0.0
    u1, s1, v1 = get_svd(mat1) if mat1 != None else (None, None, None)
    u2, s2, v2 = get_svd(mat2) if mat2 != None else (None, None, None)
    svd1_flops = flops_from_svd(u1, s1, v1)
    svd2_flops = flops_from_svd(u2, s2, v2)
    if mat1 != None and mat1.shape[0] == 1:
        svd1_flops = mat1_flops
    if mat2 != None and mat2.shape[0] == 1:
        svd2_flops = mat2_flops

    flops_per_sv_1 = 2 * v1.shape[0] + 2 * u1.shape[0] if mat1 != None else 0.0
    flops_per_sv_2 = 2 * v2.shape[0] + 2 * u2.shape[0] if mat2 != None else 0.0

    #F2_loss_1 = torch.diag(s1) if mat1 != None else []
    F2_loss_1 = [s*s for s in s1] if mat1 != None else [] 
    #F2_loss_2 = torch.diag(s2) if mat2 != None else []
    F2_loss_2 = [s*s for s in s2] if mat2 != None else []

    r1 = s1.shape[0] if mat1 != None else 0
    r2 = s2.shape[0] if mat2 != None else 0

    min_F2_loss = torch.inf

    if (mat1 == None and mat2 == None):
        print("Warning! Both matrices empty!")

    # Find the best split given that both matrices are SV-decomposed
    if (mat1 != None and mat2 != None):
        flops_both_svd = svd1_flops + svd2_flops
        F2_loss_both_svd = 0
        i1_both_svd = r1 - 1
        i2_both_svd = r2 - 1
        while flops_both_svd > flops:
            F2_per_flops_1 = F2_loss_1[i1_both_svd] / flops_per_sv_1
            F2_per_flops_2 = F2_loss_2[i2_both_svd] / flops_per_sv_2

            if (F2_per_flops_1 < F2_per_flops_2 or i2_both_svd == 0) and i1_both_svd > 0:
                F2_loss_both_svd += F2_loss_1[i1_both_svd]
                flops_both_svd -= flops_per_sv_1
                i1_both_svd -= 1
            elif i2_both_svd > 0:
                F2_loss_both_svd += F2_loss_2[i2_both_svd]
                flops_both_svd -= flops_per_sv_2
                i2_both_svd -= 1

        r1_optimal = i1_both_svd + 1
        r2_optimal = i2_both_svd + 1
        min_F2_loss = F2_loss_both_svd
        flops1 = r1_optimal * flops_per_sv_1
        flops2 = r2_optimal * flops_per_sv_2

    # Find the best split given that only mat1 is SV-decomposed
    if mat1 != None:
        flops_mat1_svd = svd1_flops + mat2_flops
        F2_loss_mat1_svd = 0
        i1_mat1_svd = r1 - 1
        while flops_mat1_svd > flops and i1_mat1_svd > 0:
            F2_loss_mat1_svd += F2_loss_1[i1_mat1_svd]
            flops_mat1_svd -= flops_per_sv_1
            i1_mat1_svd -= 1

        if i1_mat1_svd < 1:
            F2_loss_mat1_svd = torch.inf

        if F2_loss_mat1_svd <= min_F2_loss:
            r1_optimal = i1_mat1_svd + 1
            r2_optimal = r2
            min_F2_loss = F2_loss_mat1_svd
            flops1 = r1_optimal * flops_per_sv_1
            flops2 = mat2_flops

    # Find the best split given that only mat2 is SV-decomposed
    if mat2 != None:
        flops_mat2_svd = mat1_flops + svd2_flops
        F2_loss_mat2_svd = 0
        i2_mat2_svd = r2 - 1
        while flops_mat2_svd > flops and i2_mat2_svd > 0:
            F2_loss_mat2_svd += F2_loss_2[i2_mat2_svd]
            flops_mat2_svd -= flops_per_sv_2
            i2_mat2_svd -= 1

        if i2_mat2_svd < 1:
                F2_loss_mat2_svd = torch.inf

        if F2_loss_mat2_svd <= min_F2_loss:
            r1_optimal = r1
            r2_optimal = i2_mat2_svd + 1
            min_F2_loss = F2_loss_mat2_svd
            flops1 = mat1_flops
            flops2 = r2_optimal * flops_per_sv_2
    return (min_F2_loss, r1_optimal, r2_optimal, flops1, flops2)

def get_proj_loss_F2(matrix1:SubMatrix, matrix2:SubMatrix, flops:int):
    torch_matrix1 = matrix1.get_dense_torch_matrix() if matrix1.has_rows() else None
    torch_matrix2 = matrix2.get_dense_torch_matrix() if matrix2.has_rows() else None
    F2_loss, r1_optimal, r2_optimal, flops1, flops2 = find_optimal_rank_allocation_F2(torch_matrix1, torch_matrix2, flops)

    # Check if F2 loss from singular values matches actual F2 of the difference
    lora_1 = get_svd_lora(torch_matrix1, r1_optimal) if torch_matrix1 is not None else None
    lora_2 = get_svd_lora(torch_matrix2, r2_optimal) if torch_matrix2 is not None else None
    F_loss_1 = weighted_frobenious(torch_matrix1, lora_1) if torch_matrix1 is not None else 0.0
    F_loss_2 = weighted_frobenious(torch_matrix2, lora_2) if torch_matrix2 is not None else 0.0
    F2_loss_actual = F_loss_1 * F_loss_1 + F_loss_2 * F_loss_2

    if torch.abs(F2_loss - F2_loss_actual) > 0.5 and abs(F2_loss/F2_loss_actual - 1) > 0.001:
        print(f"Warning! F2 losses dont match: {F2_loss} vs {F2_loss_actual}")

    return F2_loss, r1_optimal, r2_optimal, flops1, flops2

def optimal_row_to_move_F2(sender:SubMatrix, receiver:SubMatrix, flops:int):
    # current_best_F2_loss, _, _ = get_proj_loss_F2(A_tuple_full, sender, receiver, flops)
    best_F2_loss = torch.inf
    best_row = None
    for row_i, row in sender.get_rows():
        sender_copy = sender.copy()
        receiver_copy = receiver.copy()
        sender_copy.remove_row(row_i)
        receiver_copy.add_row(row_i)
        F2_loss, _, _, _, _ = get_proj_loss_F2(sender_copy, receiver_copy, flops)
        if F2_loss < best_F2_loss:
            best_F2_loss = F2_loss
            best_row = (row_i, row)

    return best_row, best_F2_loss

def greedy_splitting_rows_F2(orig_matrix:SubMatrix, flops=None, printdepth = 1):

    # Check if group is vector
    if orig_matrix.num_rows == 1:
        print("Attempting to split a single vector. Returning vector with zero loss.")
        return 0.0, orig_matrix.get_flops_full_matrix(), [orig_matrix]

    if flops == None:
        flops = orig_matrix.get_flops_full_matrix() / 2

    print(f"Depth: {printdepth}")
    sender_idx = 1 # 1 means group1, 2 means group2
    optimal_group1, optimal_group2 = orig_matrix.get_rows(), [] # all and no rows

    current_best_F2_loss, _, _, _, _ = get_proj_loss_F2(
        SubMatrix(orig_matrix, [r[0] for r in optimal_group1]), SubMatrix(orig_matrix, [r[0] for r in optimal_group2]), flops)
    single_svd_F2_loss = current_best_F2_loss # For debugging, can remove later
    print(f"Loss from simple SVD: {single_svd_F2_loss}")

    while True:
        current_best_F2_loss_for_direction, _, _, _, _ = get_proj_loss_F2(
            SubMatrix(orig_matrix, [r[0] for r in optimal_group1]), SubMatrix(orig_matrix, [r[0] for r in optimal_group2]), flops)
        optimal_group1_for_direction = list(optimal_group1)
        optimal_group2_for_direction = list(optimal_group2)
        groups = [SubMatrix(orig_matrix, rows=[r[0] for r in optimal_group1]),
                  SubMatrix(orig_matrix, rows=[r[0] for r in optimal_group2])]
        sender = groups[sender_idx - 1]
        receiver = groups[-sender_idx]
        while sender.has_rows():
            row_to_move, F2_loss = optimal_row_to_move_F2(sender, receiver, flops)
            sender.remove_row(row_to_move[0])
            receiver.add_row(row_to_move[0])
            if F2_loss < current_best_F2_loss_for_direction:
                optimal_group1_for_direction = groups[0].get_rows()
                optimal_group2_for_direction = groups[1].get_rows()
                current_best_F2_loss_for_direction = F2_loss

        print(f"Loss for direction: {current_best_F2_loss_for_direction}")

        if current_best_F2_loss_for_direction < current_best_F2_loss:
            current_best_F2_loss = current_best_F2_loss_for_direction
            optimal_group1 = list(optimal_group1_for_direction)
            optimal_group2 = list(optimal_group2_for_direction)
            sender_idx = 2 if sender_idx == 1 else 1
            continue
        else:
            print("Optimal split found. Proceeding to split sub-matrices.")
            break
    
    optimal_matrix1 = SubMatrix(orig_matrix, [row[0] for row in optimal_group1]) 
    optimal_matrix2 = SubMatrix(orig_matrix, [row[0] for row in optimal_group2])
    optimal_group1, optimal_group2 = None, None # just to make sure we don't use them anymore

    F2_loss, r1_optimal, r2_optimal, flops1, flops2 = get_proj_loss_F2(optimal_matrix1, optimal_matrix2, flops)
    print(f"Size of group 1: {optimal_matrix1.num_rows}\nSize of group 2: {optimal_matrix2.num_rows}")
    print(f"Rank of matrix 1: {r1_optimal}\nRank of matrix 2: {r2_optimal}")

    if optimal_matrix1.has_rows() and optimal_matrix2.has_rows():
        remainderflops = flops - flops1 - flops2 # TODO: Implement efficient usage of remainder flops in another branches
        if remainderflops < 0:
            print(f"Warning: Used more flops than allocated! {flops1} + {flops2} = {flops1 + flops2} vs {flops}")
        if r2_optimal <= 1:
            flops1 += remainderflops
        elif r1_optimal <= 1:
            flops2 += remainderflops
        else:
            flops1 += int(remainderflops * 0.5)
            flops2 += remainderflops - int(remainderflops * 0.5)

        print(f"Splitting sub-matrix 1 of size {optimal_matrix1.num_rows} at depth = {printdepth} with {flops1} flops")
        if r1_optimal > 1:
            F2_loss_1, flops1_actual, groups1 = greedy_splitting_rows_F2(optimal_matrix1, flops1, printdepth + 1)
        else:
            flops1_actual = flops_from_shape(inp=optimal_matrix1.num_cols, out=optimal_matrix1.num_rows, rank=r1_optimal)
            F2_loss_1, groups1 = 0.0, [optimal_matrix1]
        print(f"Splitting sub-matrix 2 of size {optimal_matrix2.num_rows} at depth = {printdepth} with {flops2} flops")
        if r2_optimal > 1:
            F2_loss_2, flops2_actual, groups2 = greedy_splitting_rows_F2(optimal_matrix2, flops2, printdepth + 1)
        else:
            flops2_actual = flops_from_shape(inp=optimal_matrix2.num_cols, out=optimal_matrix2.num_rows, rank=r2_optimal)
            F2_loss_2, groups2 = 0.0, [optimal_matrix2]
        total_F2_loss = F2_loss_1 + F2_loss_2
        total_actual_flops = flops1_actual + flops2_actual
        return total_F2_loss, total_actual_flops, groups1 + groups2
    else:
        print("No matrices to split.")
        flops1_actual = flops1
        flops2_actual = flops2
        total_F2_loss = F2_loss
        total_actual_flops = flops1_actual + flops2_actual
        ret_group = optimal_matrix1 if optimal_matrix1.has_rows() else optimal_matrix2
        ret_group.sort_rows()
        return total_F2_loss, total_actual_flops, [ret_group]
    
def test():
    """
    If your change was a no-op and this test fails, your change broke something.
    If your change was an improvement, we probably expect an improement here too,
    if so, double change the value of the expected_loss to whatever it should be now.
    """
    def test_seed(seed, expected_loss):
        torch.manual_seed(seed)
        # torch.manual_seed(123) <- this will fail? 
        A = SubMatrix(torch.randn(50, 20))
        total_F2_loss, total_actual_flops, groups = greedy_splitting_rows_F2(A)
        assert (total_F2_loss - expected_loss).abs() <= 1e-3, f"Expected {expected_loss} loss, but instead got: {total_F2_loss} for seed {seed}"
        return total_F2_loss
    test_seed(1234, 288.4483)
    test_seed(1235, 293.4544)
    test_seed(1236, 283.0340)
    test_seed(1237, 277.2634)

if __name__ == "__main__":
    test()