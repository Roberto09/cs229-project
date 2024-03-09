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
from dataset_preprocessing import TokenInfo

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
    return keepers, keep

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

# only use most frequent tokens to fit clusters
def cluster_fit_all_layers_most_freq(K=8, percentile_train=95):
    print(f'Running pca clustering with {percentile_train} percentile tokens')
    vector_dicts_layers = get_importances()
    n_layers = len(list(vector_dicts_layers.keys()))
    
    preds_out = {i:{} for i in range(n_layers)} # layer -> token_id -> predicted cluster
    cluster_distributions = {} # layer -> cluster distribution
    cluster_freq_distributions = {} # layer -> cluster distribution by combined frequency of tokens

    for layer, v in tqdm(vector_dicts_layers.items()):
        token_freqs = np.array([x[2] for x in v.keys()]) # same for each layer
        freq_threshold = np.percentile(token_freqs, percentile_train)

        # train clustering only on top percentile tokens
        v_train = {key:val for key, val in v.items() if key[2] > freq_threshold}
        imps_train = np.array(list(v_train.values())).astype(np.float32)
        imps_all = np.array(list(v.values())).astype(np.float32)
        
        pca = PCA(n_components=0.95)
        pca.fit(imps_train)

        n_components = pca.n_components_ # for debug
        print(f'layer:{layer}\n n_components: {n_components}')
        imps_train_pca = pca.transform(imps_train)

        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(imps_train_pca)

        imps_all_pca = pca.transform(imps_all)
        cluster_preds = kmeans.predict(imps_all_pca)

        # distribution by adding up frequency of each token
        cluster_freq_distribution = [0 for _ in range(K)]

        for i, k in enumerate(v.keys()):
            pred_cluster = cluster_preds[i]
            cluster_freq_distribution[pred_cluster] += k[2] # token freq
            preds_out[layer][k[0]] = pred_cluster

        sum_freqs = sum(cluster_freq_distribution)
        freq_fractions = [round(x/sum_freqs, 4) for x in cluster_freq_distribution]
        cluster_freq_distributions[layer] = cluster_freq_distribution

        c = Counter(cluster_preds)
        cluster_distributions[layer] = sorted(c.items())
        print(f'cluster distribution: \n{cluster_distributions[layer]}')
        print(f'cluster combined freq distribution: \n{cluster_freq_distributions[layer]}')
        print(f'cluster freq fractions distribution: \n{freq_fractions}\n')

    return preds_out, cluster_distributions, cluster_freq_distributions

def cluster_fit_all_layers(K=8, train_ratio = 0.2, weighted = True):
    print(f'Running {"weighted" if weighted else "unweighted"} clustering, with random sampled tokens\n')
    vector_dicts_layers = get_importances()
    n_layers = len(list(vector_dicts_layers.keys()))
    
    preds_out = {i:{} for i in range(n_layers)} # layer -> token_id -> predicted cluster
    cluster_distributions = {} # layer -> cluster distribution
    cluster_freq_distributions = {} # layer -> cluster distribution by combined frequency of tokens

    for layer, v in tqdm(vector_dicts_layers.items()):       
        vectors_clean, _ = clean_data(v)
        vectors_clean_values = np.array(list(vectors_clean.values()))

        num_rows = vectors_clean_values.shape[0]
        idx = np.random.choice(num_rows, int(num_rows * train_ratio), replace=False)

        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        if weighted:
            token_weights = np.array([token_id[2] for token_id in v.keys()])
            kmeans.fit(vectors_clean_values[idx], sample_weight=token_weights[idx])
        else:
            kmeans.fit(vectors_clean_values[idx])

        vectors_all_values = np.array(list(v.values()))
        cluster_preds = kmeans.predict(vectors_all_values)

        # distribution by adding up frequency of each token
        cluster_freq_distribution = [0 for _ in range(K)]

        for i, k in enumerate(v.keys()):
            pred_cluster = cluster_preds[i]
            cluster_freq_distribution[pred_cluster] += k[2] # token freq
            preds_out[layer][k[0]] = pred_cluster
        
        sum_freqs = sum(cluster_freq_distribution)
        freq_fractions = [round(x/sum_freqs, 4) for x in cluster_freq_distribution]
        cluster_freq_distributions[layer] = cluster_freq_distribution
        
        c = Counter(cluster_preds)
        cluster_distributions[layer] = sorted(c.items())
        print(f'Layer: {layer}\n cluster distribution: \n{cluster_distributions[layer]}')
        print(f'cluster combined freq distribution: \n{cluster_freq_distributions[layer]}')
        print(f'cluster freq fractions distribution: \n{freq_fractions}\n')
    
    return preds_out, cluster_distributions, cluster_freq_distributions

def cluster_fit_all_layers_inputs(K=8):
    """
    fit KMeans and collect embeddings for each cluster per layer
    """
    vector_dicts_layers = get_importances_inputs()
    n_layers = len(list(vector_dicts_layers.keys()))
    token_freq = TokenInfo().token_counts # this takes a minute or two
    
    clusters_out = {} # layer -> fitted kmeans model
    embeddings_clusters_out = {i:{} for i in range(n_layers)} # layer -> cluster -> (stacked list of embeddings)
    cluster_distributions = {} # layer -> cluster distribution
    for layer, layer_dict in tqdm(vector_dicts_layers.items()):
        token_infos = layer_dict.keys() # not used in this context
        token_weights = np.array([token_freq[token_id[0]] for token_id in token_infos])
        importance_vectors = np.array([x[0] for x in layer_dict.values()])
        input_embeddings = [x[1] for x in layer_dict.values()]

        # this data is already sampled, dont need to select random indices to fit

        importance_vectors_clean, clean_idx = clean_vectors(importance_vectors)
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(importance_vectors_clean, sample_weight=token_weights[clean_idx])
        cluster_preds = kmeans.predict(importance_vectors)
        
        c = Counter(cluster_preds)
        cluster_distributions[layer] = sorted(c.items())

        for c in c.keys():
            embeddings_clusters_out[layer][c] = [] # setup, not guarenteed to have exactly k clusters

        for i, embedding in enumerate(input_embeddings):
            embeddings_clusters_out[layer][cluster_preds[i]].append(embedding)

        clusters_out[layer] = kmeans
    
    return embeddings_clusters_out, clusters_out, cluster_distributions


def dump_clusters_most_freq(percentile_train=95):
    preds, cluster_distributions, cluster_freq_distributions\
          = cluster_fit_all_layers_most_freq(percentile_train=percentile_train)
    
    with open('cluster_pkl/clustering_most_freq_preds.pkl', 'wb') as f:
        pickle.dump(preds, f)
    
    with open('cluster_pkl/clustering_most_freq_cluster_distributions.pkl', 'wb') as f:
        pickle.dump(cluster_distributions, f)
    
    with open('cluster_pkl/clustering_most_freq_cluster_freq_distributions.pkl', 'wb') as f:
        pickle.dump(cluster_freq_distributions, f)

def dump_clusters(weighted=True):
    preds, cluster_distributions, cluster_freq_distributions\
          = cluster_fit_all_layers(weighted=weighted)
    
    with open(f'cluster_pkl/clustering_{"weighted" if weighted else "unweighted"}_preds.pkl', 'wb') as f:
        pickle.dump(preds, f)
    
    with open(f'cluster_pkl/clustering_{"weighted" if weighted else "unweighted"}_cluster_distributions.pkl', 'wb') as f:
        pickle.dump(cluster_distributions, f)
    
    with open(f'cluster_pkl/clustering_{"weighted" if weighted else "unweighted"}_cluster_freq_distributions.pkl', 'wb') as f:
        pickle.dump(cluster_freq_distributions, f)


def dump_embeddings_clusters():
    embeddings_clusters_out, _, cluster_distributions = cluster_fit_all_layers_inputs()
    with open('cluster_pkl/clustering_embeddings_weighted.pkl', 'wb') as f:
        pickle.dump(embeddings_clusters_out, f)
        

if __name__ == '__main__':
    dump_clusters(weighted=False)
    dump_clusters(weighted=True)
    dump_clusters_most_freq()