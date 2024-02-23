import torch

@torch.no_grad()
def prune_single_mlp(mlp, importances, prune_ratio):
    """ Given a single mlp, it's importances and a prune ratio, it prunes it.
    """
    # sorts from least to most important
    sorted_imps_idx = torch.argsort(importances)
    num_prune_cells = int(sorted_imps_idx.shape[0] * prune_ratio)
    keep_cells = sorted_imps_idx[num_prune_cells:]
    keep_cells = torch.sort(keep_cells).values

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
    # TODO(sorvisto) do this.
    raise NotImplementedError()
