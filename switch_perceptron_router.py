import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


# this is assuming 8 experts
def init_weights_by_centroids(layer):
    with open('cluster_pkl/clustering_embeddings.pkl', 'rb') as file:
            clustered_embeddings = pickle.load(file)
            clusters = [np.array(c) for c in clustered_embeddings[layer].values()]
            centroids = np.vstack([np.mean(c, axis = 0) for c in clusters])
    
    return torch.tensor(centroids, dtype = torch.float)

# https://arxiv.org/pdf/2101.03961.pdf
class SwitchPerceptronRouter(nn.Module):
    def __init__(self, input_size, n_experts, layer):
        super().__init__()
        self.fc = nn.Linear(input_size, n_experts)
        self.fc.weight = nn.Parameter(init_weights_by_centroids(layer))
        
    def forward(self, x):
        logits = self.fc(x)
        softmax_values = F.softmax(logits, dim=1)
        expert_idx = torch.argmax(softmax_values, dim=1, keepdim=True) # index of expert to route to
        router_gate_values = torch.gather(softmax_values, 1, expert_idx)
        return expert_idx, router_gate_values # expert, reouter gate value

if __name__ == '__main__':
     layer = 12
     router = SwitchPerceptronRouter(2048, 8, layer)
     
     # test functionality in context
     with open('cluster_pkl/clustering_embeddings.pkl', 'rb') as file:
            clustered_embeddings = pickle.load(file)

     embeddings = np.vstack([np.array(c) for c in clustered_embeddings[layer].values()])
     idx = np.random.choice(embeddings.shape[0], size=50, replace=False)
     embeddings = embeddings[idx]
     embeddings = torch.tensor(embeddings, dtype = torch.float)

     experts_routed, gate_values = router.forward(embeddings)

     print()
