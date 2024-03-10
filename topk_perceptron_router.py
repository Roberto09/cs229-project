import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

# this is assuming 8 experts
def init_weights_by_centroids(layer, use_pca_clusters):
    cluster_path = f'cluster_pkl/cluster_embeddings_{"PCA" if use_pca_clusters else ""}.pkl'
    with open(cluster_path, 'rb') as file:
            clustered_embeddings = pickle.load(file)
            clusters = list(clustered_embeddings[layer].values())
            cluster_embeddings = [[x[0] for x in c] for c in clusters]
            cluster_weights = [np.array([x[1] for x in c]) for c in clusters]
            centroids = np.vstack([np.average(c, axis = 0, weights = cluster_weights[i]) for i, c in enumerate(cluster_embeddings)])
    
    return torch.tensor(centroids, dtype = torch.float)

# For switch routing(https://arxiv.org/pdf/2101.03961.pdf) use k = 1
class TopKPerceptronRouter(nn.Module):
    def __init__(self, input_size, n_experts, layer, k, cluster_init=True, use_pca_clusters=False):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(input_size, n_experts)
        if cluster_init:
            self.fc.weight = nn.Parameter(init_weights_by_centroids(layer, use_pca_clusters))
            self.fc.bias = nn.Parameter(torch.zeros(n_experts))

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(-1, feature_dim)  # Shape: [batch_size * seq_len, feature_dim]
        logits = self.fc(x)
        softmax_values = F.softmax(logits, dim=1)
        top_k_expert_weights, top_k_experts_idx = torch.topk(softmax_values, self.k, dim=1)
        
        top_k_experts_idx = top_k_experts_idx.view(batch_size, seq_len, self.k)
        top_k_expert_weights = top_k_expert_weights.view(batch_size, seq_len, self.k)
        return top_k_experts_idx, top_k_expert_weights  # Shapes: [batch_size, seq_len, k], [batch_size, seq_len, k]    
        

if __name__ == '__main__':
     # test functionality in context
     layer = 12
     with open('cluster_pkl/cluster_embeddings_.pkl', 'rb') as file:
            clustered_embeddings = pickle.load(file)
    
     clusters = list(clustered_embeddings[layer].values())
     cluster_embeddings = [[x[0] for x in c] for c in clusters]
     embeddings = np.vstack(cluster_embeddings[3]) # pick random
     idx = np.random.choice(embeddings.shape[0], size=50, replace=True)
     embeddings = embeddings[idx]
     embeddings = torch.tensor(embeddings, dtype = torch.float)

     embeddings = embeddings.unsqueeze(1)
     embeddings = embeddings.repeat(1, 192, 1)  # Shape: [batch_size, seq_len, feature_dim]
     
     router = TopKPerceptronRouter(2048, 8, layer, k=2, use_pca_clusters=True)
     experts_routed_pca, expert_weights_pca = router(embeddings)

     router = TopKPerceptronRouter(2048, 8, layer, k=1, use_pca_clusters=True)
     experts_routed_pca_, expert_weights_pca_ = router(embeddings)

     router = TopKPerceptronRouter(2048, 8, layer, k=2, use_pca_clusters=False)
     experts_routed, expert_weights = router(embeddings)

     router = TopKPerceptronRouter(2048, 8, layer, k=1, use_pca_clusters=False)
     experts_routed_, expert_weights_ = router(embeddings)
     
     print()
