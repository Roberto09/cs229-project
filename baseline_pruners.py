import torch

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
