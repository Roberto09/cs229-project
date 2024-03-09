import pickle
from torch import nn, zeros, tensor, int
from dataset_preprocessing import TokenInfo # for test
import torch

class ClusterRouter(nn.Module):
    def __init__(self, layer, vocab_size, n_experts):
        # layer (int): zero-indexed layer number
        # vocac_size (int): size of tokenizer vocabulary, number of unique tokens
        super().__init__()

        with open('cluster_pkl/clustering_most_freq_preds.pkl', 'rb') as file:
            pred_dicts = pickle.load(file)
            cluster_preds_layer = pred_dicts[layer]
        
        router = zeros(vocab_size, dtype=int)
        self.register_buffer('router', router)
        # self.register_buffer('cnt', torch.zeros(n_experts, dtype=int))

        for i in range(vocab_size):
            if i in cluster_preds_layer.keys():
                self.router[i] = cluster_preds_layer[i]
            else:
                self.router[i] = i % n_experts
        
    def forward(self, x):
        # x (tensor 2d): token ids
        res = self.router[x]
        # res_s = res.view(-1)
        # eos_tid = 50256
        # self.cnt += torch.bincount(res_s[x.view(-1) != eos_tid])
        return res

if __name__ == '__main__':
    # test funcionality
    cluster_router = ClusterRouter(0, 52000, 8) # not exact size of vocab
    token_info = TokenInfo()
    tokens_all = token_info.top_n(1000)
    examples = token_info.get_prefixes(tokens_all[999][0], 10, 10)
    examples = tensor(examples, device="cpu") # TODO: change to cuda
    
    experts_routed = cluster_router.forward(examples)

    print()