import torch
from tqdm import tqdm

@torch.no_grad()
def get_mlp_loras_simple(mlp, most_imp_cells, rank=64):
    """
    Given an MLP, the indices of cells (most_imp_cells) and a rank, it returns simple
    initializations for the A,B low-rank matrices of fc1 and fc2.
    """
    print("get_mlp_loras_simple")
    # assumes descending order of importances in most_imp_cells
    fc1_mat = mlp.fc1.weight[most_imp_cells[:rank]]
    fc1_bias = mlp.fc1.bias[most_imp_cells] # bias is full dimension
    fc2_mat = mlp.fc2.weight[:, most_imp_cells[:rank]]
    # no need of fc2 bias
    
    fc1_A = fc1_mat
    fc1_B = torch.concat((torch.eye(rank), torch.zeros(len(most_imp_cells) - rank, rank)), dim=0)
    
    fc2_B = fc2_mat
    fc2_A = torch.concat((torch.eye(rank), torch.zeros(rank, len(most_imp_cells) - rank)), dim=1)
    # fc2_A = fc2_mat
    # fc2_B = torch.concat((torch.eye(rank), torch.zeros(mlp.fc2.out_features - rank, rank)), dim=0)
    return fc1_A.clone(), fc1_B.clone(), fc1_bias.clone(), fc2_B.clone(), fc2_A.clone()

class SVDRes():
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

def get_svd(tens):
    # print("running svd")
    tens = tens.cuda()
    res = torch.svd(tens)
    return SVDRes(res.U.cpu(), res.S.cpu(), res.V.cpu())

@torch.no_grad()
def get_mlp_loras_svd(mlp, most_imp_cells, rank=64, init_bias=True):
    """
    Given an MLP, the indices of cells (most_imp_cells) and a rank, it returns SVD
    initializations for the A,B low-rank matrices of fc1 and fc2.
    """
    print("get_mlp_loras_svd")
    # assumes descending order of importances in most_imp_cells
    fc1_mat = mlp.fc1.weight[most_imp_cells]
    if init_bias:
        fc1_bias = mlp.fc1.bias[most_imp_cells]
    else:
        assert False, "not implemented yet"
    fc2_mat = mlp.fc2.weight[:, most_imp_cells]
    # no need of fc2 bias

    fc1_svd = get_svd(fc1_mat)
    fc2_svd = get_svd(fc2_mat)
    
    fc1_B = fc1_svd.U[:, :rank] @ torch.diag(fc1_svd.S[:rank])
    fc1_A = fc1_svd.V.T[:rank]

    fc2_B = fc2_svd.U[:, :rank] @ torch.diag(fc2_svd.S[:rank])
    fc2_A = fc2_svd.V.T[:rank]
    return fc1_A.clone(), fc1_B.clone(), fc1_bias.clone(), fc2_B.clone(), fc2_A.clone()

@torch.no_grad()
def initialize_loras(orig_model, model, init_cells, lora_func=get_mlp_loras_simple):
    """
    Given the original model, the model with experts, and the hidden cells for each MLP
    that we should be initializing, it initializes the cells indicated by the indices in
    init_cells to their SVD.
    """
    for layer_i, (orig_layer, layer) in tqdm(enumerate(zip(orig_model.model.layers, model.model.layers))):
        num_experts = len(init_cells[layer_i])
        for expert_i in range(num_experts): # 
            most_imp_cells = init_cells[layer_i][expert_i]
            rank = layer.mlp.experts_fc1[expert_i].orig_lora.lora_A.default.out_features
            print(f"using rank {rank}")
            assert rank == layer.mlp.experts_fc1[expert_i].orig_lora.lora_A.default.weight.shape[0]
            
            fc1_A, fc1_B, fc1_bias, fc2_B, fc2_A = lora_func(orig_layer.mlp, most_imp_cells, rank)

            layer.mlp.experts_fc1[expert_i].orig_lora.lora_A.default.weight.data = fc1_A
            layer.mlp.experts_fc1[expert_i].orig_lora.lora_B.default.weight.data = fc1_B
            layer.mlp.experts_fc1[expert_i].lora_bias.data = fc1_bias
            
            layer.mlp.experts_fc2[expert_i].orig_lora.lora_A.default.weight.data = fc2_A
            layer.mlp.experts_fc2[expert_i].orig_lora.lora_B.default.weight.data = fc2_B
            # no bias needed for fc2
