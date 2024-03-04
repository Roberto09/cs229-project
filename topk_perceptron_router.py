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

# For switch routing(https://arxiv.org/pdf/2101.03961.pdf) use k = 1
class TopKPerceptronRouter(nn.Module):
    def __init__(self, input_size, n_experts, layer, k):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(input_size, n_experts)
        self.fc.weight = nn.Parameter(init_weights_by_centroids(layer))
        
    def forward(self, x):
        logits = self.fc(x)
        softmax_values = F.softmax(logits, dim=1)
        top_k_expert_weights, top_k_experts_idx = torch.topk(softmax_values, self.k, dim=1)
        return top_k_experts_idx, top_k_expert_weights # expert indices, expert_weights

if __name__ == '__main__':
     # test functionality in context
     layer = 12
     with open('cluster_pkl/clustering_embeddings.pkl', 'rb') as file:
            clustered_embeddings = pickle.load(file)

     embeddings = np.vstack([np.array(c) for c in clustered_embeddings[layer].values()])
     idx = np.random.choice(embeddings.shape[0], size=50, replace=False)
     embeddings = embeddings[idx]
     embeddings = torch.tensor(embeddings, dtype = torch.float)
     
     router = TopKPerceptronRouter(2048, 8, layer, k=2)
     experts_routed, expert_weights = router.forward(embeddings)

     router = TopKPerceptronRouter(2048, 8, layer, k=1)
     experts_routed_, expert_weights_ = router.forward(embeddings)
     
     print()
