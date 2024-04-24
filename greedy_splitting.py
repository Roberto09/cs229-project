import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import pandas as pd
import numpy as np

from weighted_svd_utils import weighted_frobenious, get_svd, rank_r, get_svd_lora, toy_correlated_matrix, toy_two_subspace_matrix, toy_seven_subspace_matrix, toy_seven_subspace_matrix_unqual_size

def flops_from_svd(U, S, V):
    ret = 2 * (V.shape[0] * V.shape[1] + U.shape[0] * U.shape[1]) if U != None and S != None and V != None else 0
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

def get_proj_loss_F2(group1, group2, flops):

    mat1 = torch.stack([row for idx, row in group1]) if group1 else None
    mat2 = torch.stack([row for idx, row in group2]) if group2 else None

    F2_loss, r1_optimal, r2_optimal, flops1, flops2 = find_optimal_rank_allocation_F2(mat1, mat2, flops)

    # Check if F2 loss from singular values matches actual F2 of the difference
    lora_1 = get_svd_lora(mat1, r1_optimal) if group1 else None
    lora_2 = get_svd_lora(mat2, r2_optimal) if group2 else None
    A_partial_1 = torch.stack([row for idx, row in group1]) if group1 else None
    A_partial_2 = torch.stack([row for idx, row in group2]) if group2 else None
    F_loss_1 = weighted_frobenious(A_partial_1, lora_1) if group1 else 0.0
    F_loss_2 = weighted_frobenious(A_partial_2, lora_2) if group2 else 0.0
    F2_loss_actual = F_loss_1 * F_loss_1 + F_loss_2 * F_loss_2

    if torch.abs(F2_loss - F2_loss_actual) > 0.5 and abs(F2_loss/F2_loss_actual - 1) > 0.001:
        print(f"Warning! F2 losses dont match: {F2_loss} vs {F2_loss_actual}")

    return F2_loss, r1_optimal, r2_optimal, flops1, flops2

def optimal_row_to_move_F2(sender, receiver, flops):
    # current_best_F2_loss, _, _ = get_proj_loss_F2(A_tuple_full, sender, receiver, flops)
    current_best_F2_loss = torch.inf
    current_best_tuple = None
    for tuple in sender:
        s = list(sender)
        r = list(receiver)

        r.append(tuple)
        s.remove(tuple)
        F2_loss, _, _, _, _ = get_proj_loss_F2(s, r, flops)
        if F2_loss < current_best_F2_loss:
            current_best_F2_loss = F2_loss
            current_best_tuple = tuple

    return current_best_tuple, current_best_F2_loss

def greedy_splitting_rows_F2(row_tuples, flops=None, printdepth = 1):

    # Check if group is vector
    if len(row_tuples) == 1:
        print("Attempting to split a single vector. Returning vector with zero loss.")
        return 0.0, 2 * len(row_tuples) * len((row_tuples[0])[1]), [row_tuples]

    if flops == None:
        A_flops = 2 * len(row_tuples) * len((row_tuples[0])[1])
        flops = 0.5 * A_flops

    print(f"Depth: {printdepth}")

    sender_idx = 1 # 1 means group1, 2 means group2
    optimal_group1 = list(row_tuples)
    optimal_group2 = []

    current_best_F2_loss, _, _, _, _ = get_proj_loss_F2(optimal_group1, optimal_group2, flops)
    single_svd_F2_loss = current_best_F2_loss # For debugging, can remove later
    print(f"Loss from simple SVD: {single_svd_F2_loss}")

    while True:
        current_best_F2_loss_for_direction, _, _, _, _ = get_proj_loss_F2(optimal_group1, optimal_group2, flops)
        optimal_group1_for_direction = list(optimal_group1)
        optimal_group2_for_direction = list(optimal_group2)
        groups = [list(optimal_group1), list(optimal_group2)]
        sender = groups[sender_idx - 1]
        receiver = groups[-sender_idx]
        while sender:
            row_to_move, F2_loss = optimal_row_to_move_F2(sender, receiver, flops)
            sender.remove(row_to_move)
            receiver.append(row_to_move)
            if F2_loss < current_best_F2_loss_for_direction:
                optimal_group1_for_direction = list(groups[0])
                optimal_group2_for_direction = list(groups[1])
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

    F2_loss, r1_optimal, r2_optimal, flops1, flops2 = get_proj_loss_F2(optimal_group1, optimal_group2, flops)
    print(f"Size of group 1: {len(optimal_group1)}\nSize of group 2: {len(optimal_group2)}")
    print(f"Rank of matrix 1: {r1_optimal}\nRank of matrix 2: {r2_optimal}")

    if optimal_group1 and optimal_group2:
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

        #mat1 = torch.stack([row for idx, row in optimal_group1])
        #mat2 = torch.stack([row for idx, row in optimal_group2])
        print(f"Splitting sub-matrix 1 of size {len(optimal_group1)} at depth = {printdepth} with {flops1} flops")
        if r1_optimal > 1:
            F2_loss_1, flops1_actual, groups1 = greedy_splitting_rows_F2(optimal_group1, flops1, printdepth + 1)
        else:
            F2_loss_1, flops1_actual, groups1 = 0.0, 2 * len((optimal_group1[0])[1]) + len(optimal_group1), optimal_group1
        print(f"Splitting sub-matrix 2 of size {len(optimal_group2)} at depth = {printdepth} with {flops2} flops")
        if r2_optimal > 1:
            F2_loss_2, flops2_actual, groups2 = greedy_splitting_rows_F2(optimal_group2, flops2, printdepth + 1)
        else:
            F2_loss_2, flops2_actual, groups2 = 0.0, 2 * len((optimal_group2[0])[1]) + len(optimal_group2), optimal_group2
        total_F2_loss = F2_loss_1 + F2_loss_2
        total_actual_flops = flops1_actual + flops2_actual
        return total_F2_loss, total_actual_flops, groups1 + groups2
    else:
        print("No matrices to split.")
        flops1_actual = flops1
        flops2_actual = flops2
        total_F2_loss = F2_loss
        total_actual_flops = flops1_actual + flops2_actual
        ret_group = optimal_group1 if optimal_group1 else optimal_group2
        ret_group.sort(key=lambda x: x[0])
        return total_F2_loss, total_actual_flops, [ret_group]
    
def test():
    torch.manual_seed(1234)
    # torch.manual_seed(123) <- this will fail? 
    A = torch.randn(50, 20)
    A_rows = list(enumerate(A))
    total_F2_loss, total_actual_flops, groups = greedy_splitting_rows_F2(A_rows)
    expected_loss = 288.4483
    assert (total_F2_loss - expected_loss).abs() <= 1e-3, f"Expected {expected_loss} loss, but instead got: {total_F2_loss}."


if __name__ == "__main__":
    test()