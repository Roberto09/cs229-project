import torch
import numpy as np

@torch.no_grad()
def prune_single_mlp(mlp, importances, prune_ratio):
    """ Given a single mlp, it's importances and a prune ratio, it prunes it.
    """
    # sorts from least to most important
    sorted_imps_idx = torch.argsort(importances)
    num_prune_cells = int(sorted_imps_idx.shape[0] * prune_ratio)
    keep_cells = sorted_imps_idx[num_prune_cells:]
    prune_cells = sorted_imps_idx[:num_prune_cells]
    keep_cells = torch.sort(keep_cells).values # why sort and why call values() here?

    fc1 = mlp.fc1
    dtype = fc1.weight.dtype
    fc1_pruned = torch.nn.Linear(
        fc1.weight.shape[1],
        keep_cells.shape[0],
        dtype=dtype)
    with torch.no_grad():
        fc1_pruned.weight.data = torch.clone(fc1.weight[keep_cells])
        fc1_pruned.bias.data = torch.clone(fc1.bias[keep_cells])

    fc2 = mlp.fc2
    fc2_pruned = torch.nn.Linear(
        keep_cells.shape[0],
        fc2.weight.shape[0],
        dtype=dtype)
    with torch.no_grad():
        fc2_pruned.weight.data = torch.clone(fc2.weight[:, keep_cells])
    
    mlp.fc1 = fc1_pruned
    mlp.fc2 = fc2_pruned
    return prune_cells


@torch.no_grad()
def prune_mlps_individually(importances, prune_ratio):
    """ Given a dictionary of mlp -> importance tensor, prunes
    each mlp individually to the specified prune ratio.
    Returns a list with the pruned away indexes of the cells.
    """
    prune_cells = []
    for mlp, imp in importances.items():
        mlp_prune_cells = prune_single_mlp(mlp, imp, prune_ratio)
        prune_cells.append(mlp_prune_cells.numpy())
    return prune_cells


def get_r(prune_ratio):
    gamma=prune_ratio
    h = 8192
    i = 2048
    keep_cost = 4*(1-gamma)*h*i + 2*(1-gamma)*h + i
    reg_cost = 4*0.5*h*i + 2*0.5*h + i

    r = (reg_cost - keep_cost - (2*gamma*h + i)) / (4*(gamma*h + i))
    return r


@torch.no_grad()
def prune_mlps_holistically(importances, prune_ratio, extra_prune_ratio):
    """ Given a dictionary of mlp -> importance tensor, prunes
    all the mlps holistically.
    """

    # Concatenate all importance tensors
    concat_imps = torch.cat(list(importances.values())).float()

    num_prune_cells = int(len(concat_imps) * prune_ratio)

    # Choose which node-indexes to prune, mark those indexes with '0'
    _, indices_to_replace = torch.topk(concat_imps, num_prune_cells, largest=False)
    mask = torch.ones_like(concat_imps, dtype=torch.bool)
    mask[indices_to_replace] = False
    
    # Make a new dict with indexes with smallest values zeroed out
    split_size = len(list(importances.values())[0])

    pruned_masks = torch.split(mask, split_size)
    pruned_masks_dict = {key: tensor for key, tensor in zip(importances.keys(), pruned_masks)}

    pruned_idx_list = []
    r_list = []

    # Prune each mlp
    i = 0
    for mlp, mask in pruned_masks_dict.items():
        keep_idx = torch.arange(mask.shape[0], dtype=torch.long)[mask]
        imps = list(importances.values())[i]
        imps = imps.float()
        _, keep_idx = torch.topk(imps, int(len(keep_idx)*extra_prune_ratio), largest=True)
        # keep_idx = torch.arange(keep_idx.shape[0], dtype=torch.long)[keep_idx]
        mask_prune = torch.ones_like(imps, dtype=torch.bool)
        mask_prune[keep_idx] = False
        prune_idx = torch.arange(len(list(importances.values())[i]))[mask_prune]
        i += 1
        pruned_idx_list.append(np.array(prune_idx, dtype=float))
        r_list.append(get_r(len(prune_idx) / 8192))

        fc1 = mlp.fc1
        dtype = fc1.weight.dtype
        fc1_pruned = torch.nn.Linear(
            fc1.weight.shape[1],
            keep_idx.shape[0],
            dtype=dtype
        )
        with torch.no_grad():
            fc1_pruned.weight.data = torch.clone(fc1.weight[keep_idx])
            fc1_pruned.bias.data = torch.clone(fc1.bias[keep_idx])

        fc2 = mlp.fc2
        dtype = fc2.weight.dtype
        fc2_pruned = torch.nn.Linear(
            keep_idx.shape[0],
            fc2.weight.shape[0],
            dtype=dtype
        )
        with torch.no_grad():
            fc2_pruned.weight.data = torch.clone(fc2.weight[:, keep_idx])

        mlp.fc1 = fc1_pruned
        mlp.fc2 = fc2_pruned


    return pruned_idx_list, r_list


@torch.no_grad()
def prune_mlp_random(mlp, prune_ratio):
    """
        Prune random cells
    """
    fc1 = mlp.fc1
    dtype = fc1.weight.dtype
    num_prune_cells = int(fc1.weight.shape[0] * prune_ratio)
    keep_cells = torch.randperm(fc1.weight.shape[0])[num_prune_cells:]
    
    fc1_pruned = torch.nn.Linear(
        fc1.weight.shape[1],
        keep_cells.shape[0],
        dtype=dtype)
    with torch.no_grad():
        fc1_pruned.weight.data = torch.clone(fc1.weight[keep_cells])
        fc1_pruned.bias.data = torch.clone(fc1.bias[keep_cells])

    fc2 = mlp.fc2
    fc2_pruned = torch.nn.Linear(
        keep_cells.shape[0],
        fc2.weight.shape[0],
        dtype=dtype)
    with torch.no_grad():
        fc2_pruned.weight.data = torch.clone(fc2.weight[:, keep_cells])
    
    mlp.fc1 = fc1_pruned
    mlp.fc2 = fc2_pruned

@torch.no_grad()
def prune_mlp_magnitude(mlp, prune_ratio):
    """ Given a single mlp, prune cells by magnitude of incoming weights
        This is a naive approach since we are pruning entire cells, not individual weights
    """
    fc1 = mlp.fc1
    dtype = fc1.weight.dtype
    num_prune_cells = int(fc1.weight.shape[0] * prune_ratio)
    row_norms = torch.norm(fc1.weight, p=1, dim=1)
    col_norms = torch.norm(fc2.weight.T, p=1, dim=1)
    l1_imps = row_norms + col_norms
    sorted_magnitude_idx = torch.argsort(row_norms)

    keep_cells = sorted_magnitude_idx[num_prune_cells:]
    keep_cells = torch.sort(keep_cells).values # why sort and why call values() here?

    fc1_pruned = torch.nn.Linear(
        fc1.weight.shape[1],
        keep_cells.shape[0],
        dtype=dtype)
    with torch.no_grad():
        fc1_pruned.weight.data = torch.clone(fc1.weight[keep_cells])
        fc1_pruned.bias.data = torch.clone(fc1.bias[keep_cells])

    fc2 = mlp.fc2
    fc2_pruned = torch.nn.Linear(
        keep_cells.shape[0],
        fc2.weight.shape[0],
        dtype=dtype)
    with torch.no_grad():
        fc2_pruned.weight.data = torch.clone(fc2.weight[:, keep_cells])
    
    mlp.fc1 = fc1_pruned
    mlp.fc2 = fc2_pruned


@torch.no_grad()
def prune_mlps(mlps, prune_ratio, pruning_fn):
    for mlp in mlps:
        pruning_fn(mlp, prune_ratio)