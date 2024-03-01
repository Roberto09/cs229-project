from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import os
import pandas as pd
import pickle

def compute_clustering(data_dict, k):
    """ Runs clustering on the given data dict, which is a dictionary
    of str -> np.array mapping a token to it's importances.
    """
    if isinstance(data_dict, dict):
        vectors = np.array(list(data_dict.values()))
    else:
        vectors = data_dict
    
    n = vectors.shape[0]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(vectors)

    # R^2
    inertia = kmeans.inertia_
    inertia_avg = inertia / n

    global_average = vectors.sum(axis=0) / n
    global_distance_squared_avg = np.sum(((vectors - global_average)**2)) / n

    R2 = 1 - inertia_avg / global_distance_squared_avg
    
    # Average importance distribution
    average_importance_dist = vectors.sum(axis=0) / n

    # Average ordered importance distribution
    sorted_vectors = np.sort(vectors, axis=1)[:, ::-1]
    average_ordered_importance_dist = sorted_vectors.sum(axis=0) / n

    return R2, average_importance_dist, kmeans, average_ordered_importance_dist

def plot_elbow(data_dict, k_range):
    """Plots elbow given datadict."""
    inertias_normalized = []
    for k in tqdm(k_range):
        _, __, kmeans, _ = compute_clustering(data_dict, k)
        n = len(data_dict)
        print(n)
        inertia_normalized = kmeans.inertia_ / n
        inertias_normalized.append(inertia_normalized)
        c = Counter(kmeans.labels_)
        print(f'Cluster distribution:\n {sorted(c.items())}')

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias_normalized, '-o', label='Normalized Inertia')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Normalized Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.legend()
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

def permute_vectors(vectors_shuffled):    
    for i in range(vectors_shuffled.shape[1]):
        rand_perm = np.random.permutation(np.arange(vectors_shuffled.shape[0]))
        vectors_shuffled[:,i] = vectors_shuffled[:,i][rand_perm]
    return vectors_shuffled 

def permutation_test(data_dict, k, n_perm):
    """Takes every column of the dataset and permute's it's values.
    Further it returns a list of n_perm R^2 obtained by doing the permutation
    and running clustering on the permuted datset."""
    vectors = np.array(list(data_dict.values()))
    R2s = []
    for perm in tqdm(range(n_perm)):
        vectors_shuffled = vectors.copy()
        vectors_shuffled = permute_vectors(vectors_shuffled)
        R2, _, kmeans, _ = compute_clustering(vectors_shuffled, k=k)
        R2s.append((R2, kmeans))
    return R2s

def clean_data(data_dict, drop=0.05):
    """ Cleans up datadict by dropping outliers, which are defined
    as those furthest from the centroid of the datadict.
    """
    data = list(data_dict.items())
    vectors = np.array([d[1] for d in data])
    dists = ((vectors - vectors.mean(axis=0))**2).sum(axis=1)
    drop = int(drop * len(vectors))
    
    partition = np.argpartition(dists,-drop)
    keep = partition[:-drop]
    drop = partition[-drop:]
    
    keepers = [data[i] for i in keep]
    drop = [data[i] for i in drop]
    return dict(keepers), dict(drop)

def clean_vectors(vectors, drop=0.05):
    """ Cleans up vectors by dropping outliers, which are defined
    as those furthest from the centroid of the datadict.
    """
    dists = ((vectors - vectors.mean(axis=0))**2).sum(axis=1)
    drop = int(drop * len(vectors))
    
    partition = np.argpartition(dists,-drop)
    keep = partition[:-drop]
    drop = partition[-drop:]
    
    keepers = [vectors[i] for i in keep]
    return keepers

def get_groups(data_dict, kmeans):
    """ Given a datadict and it's kmeans, returns it's groups.
    """
    toks = list(data_dict.keys())
    labels = kmeans.labels_.tolist()
    groups = {}
    for tok, label in zip(toks, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(tok)
    return groups

def visualize_kmeans(vectors, decomp = PCA, k = 8):
    """Runs kmeans over the given vectors and provides a 3d visualization."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(vectors)
    
    decomp_ = decomp(n_components=3, random_state=42)
    vec_decomp = decomp_.fit_transform(vectors)

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    colors = ['blue', 'green', 'red', 'black', 'yellow', 'purple', 'pink', 'brown']

    if k <= 8:
        scatter = ax.scatter(vec_decomp[:, 0], vec_decomp[:, 1], vec_decomp[:, 2], cmap=ListedColormap(colors[:k]), c=kmeans_labels, s=10)
        plt.colorbar(scatter, ticks=list(range(k)), label='Cluster')
    else:
        scatter = ax.scatter(vec_decomp[:, 0], vec_decomp[:, 1], vec_decomp[:, 2], c=kmeans_labels, s=10)
    
    plt.title('K-means Clusters')
    plt.show()

    silhouette_avg = silhouette_score(vectors, kmeans_labels)
    print(f'K={k}, Average Silhouette score: {silhouette_avg}\n')

    c = Counter(kmeans_labels)
    print(f'Cluster distribution:\n {sorted(c.items())}')
    
    return kmeans_labels

# Get precomputed importances
def get_importances(dir='new_importances_data'):
    vector_dict = {}
    for file in tqdm(os.listdir(dir)):
        partial_vector_dict = pd.read_pickle(os.path.join(dir, file))
        vector_dict.update(partial_vector_dict)

    n_layers = len(vector_dict[list(vector_dict.keys())[0]])
    vector_dicts_layers = {i:{k:v[i].numpy() for k,v in vector_dict.items()} for i in range(n_layers)}   
    
    return vector_dicts_layers

def get_importances_inputs(dir='importances_inputs'):
    """
    get importances and sampled embeddings(mlp inputs)
    """
    vector_dict = {}
    for file in tqdm(os.listdir(dir)):
        partial_vector_dict = pd.read_pickle(os.path.join(dir, file))
        vector_dict.update(partial_vector_dict)

    n_layers = len(vector_dict[list(vector_dict.keys())[0]][0])
    vector_dicts_layers = {i:{k:(v[0][i].numpy(), v[1][i].numpy()) for k,v in vector_dict.items()} for i in range(n_layers)}   
    
    return vector_dicts_layers

def cluster_fit_all_layers(K=8, train_ratio = 0.2):
    vector_dicts_layers = get_importances()
    n_layers = len(list(vector_dicts_layers.keys()))
    
    clusters_out = {} # layer -> fitted kmeans model
    preds_out = {i:{} for i in range(n_layers)} # layer -> token_id -> predicted cluster
    cluster_distributions = {} # layer -> cluster distribution
    for layer, v in tqdm(vector_dicts_layers.items()):
        vectors_clean, _ = clean_data(v)
        vectors_clean_values = np.array(list(vectors_clean.values()))

        num_rows = vectors_clean_values.shape[0]
        idx = np.random.choice(num_rows, int(num_rows * train_ratio), replace=False)

        # Maybe do PCA before clustering, however we didn't do that for
        # the feasibility experiment
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(vectors_clean_values[idx])

        vectors_all_values = np.array(list(v.values()))
        cluster_preds = kmeans.predict(vectors_all_values)
        
        c = Counter(cluster_preds)
        cluster_distributions[layer] = sorted(c.items())

        for i, k in enumerate(v.keys()):
            preds_out[layer][k[0]] = cluster_preds[i]

        clusters_out[layer] = kmeans
    
    return preds_out, clusters_out, cluster_distributions

def cluster_fit_all_layers_inputs(K=8):
    """
    fit KMeans and collect embeddings for each cluster per layer
    """
    vector_dicts_layers = get_importances_inputs()
    n_layers = len(list(vector_dicts_layers.keys()))
    
    clusters_out = {} # layer -> fitted kmeans model
    embeddings_clusters_out = {i:{} for i in range(n_layers)} # layer -> cluster -> (stacked list of embeddings)
    cluster_distributions = {} # layer -> cluster distribution
    for layer, layer_dict in tqdm(vector_dicts_layers.items()):
        token_infos = layer_dict.keys() # not used in this context
        importance_vectors = np.array([x[0] for x in layer_dict.values()]) # impor
        input_embeddings = [x[1] for x in layer_dict.values()]

        importance_vectors_clean = clean_vectors(importance_vectors)
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(importance_vectors_clean)
        cluster_preds = kmeans.predict(importance_vectors)
        
        c = Counter(cluster_preds)
        cluster_distributions[layer] = sorted(c.items())

        for c in c.keys():
            embeddings_clusters_out[layer][c] = [] # setup, not guarenteed to have exactly k clusters

        for i, embedding in enumerate(input_embeddings):
            embeddings_clusters_out[layer][cluster_preds[i]].append(embedding)

        clusters_out[layer] = kmeans
    
    return embeddings_clusters_out, clusters_out, cluster_distributions

def dump_clusters(train_ratio=0.2):
    preds, clusters_models, _ = cluster_fit_all_layers(train_ratio=train_ratio)
    with open('cluster_pkl/clustering_models.pkl', 'wb') as f:
        pickle.dump(clusters_models, f)
    with open('cluster_pkl/clustering_preds.pkl', 'wb') as f:
        pickle.dump(preds, f)

def dump_embeddings_clusters():
    embeddings_clusters_out, _, _ = cluster_fit_all_layers_inputs()
    with open('cluster_pkl/clustering_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_clusters_out, f)
        

if __name__ == '__main__':
    #vector_dicts_layers = get_importances()
    #preds, clusters_models, cluster_distributions = cluster_fit_all_layers()
    #dump_clusters()

    dump_embeddings_clusters()  