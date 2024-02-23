import pickle
from clustering import cluster_fit_all_layers

def dump_clusters():
    preds, clusters_models, _ = cluster_fit_all_layers()
    with open('cluster_pkl/clustering_models.pkl', 'wb') as f:
        pickle.dump(clusters_models, f)
    with open('cluster_pkl/clustering_preds.pkl', 'wb') as f:
        pickle.dump(preds, f)

if __name__ == '__main__':
    dump_clusters()