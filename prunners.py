import torch

@torch.no_grad()
def prune_single_mlp(mlp, importances, prune_ratio):
    """ Given a single mlp, it's importances and a prune ratio, it prunes it.
    """
    # sorts from least to most important
    sorted_imps_idx = torch.argsort(importances)
    num_prune_cells = int(sorted_imps_idx.shape[0] * prune_ratio)
    keep_cells = sorted_imps_idx[num_prune_cells:]
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


@torch.no_grad()
def prune_mlps_individually(importances, prune_ratio):
    """ Given a dictionary of mlp -> importance tensor, prunes
    each mlp individually to the specified prune ratio.
    """
    for mlp, imp in importances.items():
        prune_single_mlp(mlp, imp, prune_ratio)


@torch.no_grad()
def prune_mlps_holistically(importances, prune_ratio):
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

    pruned_tensors = torch.split(mask, split_size)
    pruned_tensor_dict = {key: tensor for key, tensor in zip(importances.keys(), pruned_tensors)}

    # Prune each mlp
    for mlp, keep_idx in pruned_tensor_dict.items():
        keep_idx = torch.arange(keep_idx.shape[0], dtype=torch.long)[keep_idx]
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
