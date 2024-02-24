from dataset_preprocessing import TokenInfo
from cluster_routing import ClusterRouting
import torch

# NOTE: this is just first draft, only contains routing for precomputed cluster preds
class Model():

    def __init__(self, pruned_model):
        # pruned_model: some kind of torch/huggingface pruned mode
        self.pruned_model = pruned_model
        self.cluster_router = ClusterRouting()
    

    def route_clusters(self, input_tokens, layer):
        # input tokens # (batch size, context window)
        
        # TODO: remove try cathch and change to lambda, this is just
        # because not all tokens are in the importance dict currently
        def route(id_token):
            try:
                return self.cluster_router.route(id_token, layer)
            except:
                return 0
        
        #route = lambda id_token: self.cluster_router.route(id_token, layer)
        return input_tokens.apply_(route)



if __name__ == '__main__':
    # test functionality of routing
    token_info = TokenInfo()
    tokens_all = token_info.top_n(1000)
    examples = token_info.get_prefixes(tokens_all[999][0], 10, 10)

    examples = torch.tensor(examples, device="cpu") # TODO: change to cuda

    model = Model(None)
    expert_routed = model.route_clusters(examples, 0)

    print('...')