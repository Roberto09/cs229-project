import pickle

class ClusterRouting():
    def __init__(self):
        with open('cluster_pkl/clustering_preds.pkl', 'rb') as file:
            self.cluster_preds = pickle.load(file)
    
    # both layers and experts are 0-indexed
    def route(self, id_token, layer):
        return self.cluster_preds[layer][id_token]
    

if __name__ == '__main__':
    # test funcionality
    cluster_router = ClusterRouting()
    expert = cluster_router.route(2097, 0)
    print(expert)