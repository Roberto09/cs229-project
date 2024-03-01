import torch
import itertools

def get_mlps(model):
    """Given a phi model, returns the MLP layers"""
    layers = model.get_submodule("model").get_submodule("layers")
    return [layer.get_submodule("mlp") for layer in layers]

def custom_loss(logits, labels, model):
    """ Returns crossentropy loss per token, w/o reduction """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    orig_shape = shift_labels.shape
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels).view(orig_shape)
    return loss

def compute_acc_grad(model, examples, mlps):
    """ Computes squared gradient term in delta loss approximation.
    Then it stores it in the param.acc_grad attribute."""
    params = [list(mlp.parameters()) for mlp in mlps]
    params = itertools.chain.from_iterable(params) # flatten list
    res = model(examples, labels=examples)
    losses_tens = custom_loss(res.logits, examples, model)
    losses = [loss.mean() for loss in losses_tens]
    # import pdb; pdb.set_trace()
    for example_loss in losses:
        example_loss.backward(retain_graph=True)
        for param in params: # for all the weights
            num_examples = examples.shape[0]
            with torch.no_grad():
                grad = param.grad.detach()
                sq_grad = grad * grad / num_examples
                if hasattr(param, "acc_grad"):
                    param.acc_grad += sq_grad
                else:
                    param.acc_grad = sq_grad
        model.zero_grad()
        del example_loss
        torch.cuda.empty_cache()
    return losses_tens

@torch.no_grad()
def compute_mlp_importance(mlps):
    """Given MLPS with gradients and squared gradients stored,
    approximated the importances as the delta of the loss using taylor
    expansion."""
    importances = []
    for mlp in mlps:
        fc1 = mlp.fc1
        fc2 = mlp.fc2
        # compute importance of inputs to hidden
        salience_w1 = fc1.weight * fc1.weight.grad
        salience_w2 = fc2.weight * fc2.weight.grad
        
        salience_w1 = salience_w1 - 0.5 * fc1.weight * fc1.weight.acc_grad * fc1.weight
        salience_w2 = salience_w2 - 0.5 * fc2.weight * fc2.weight.acc_grad * fc2.weight

        importance_w1_component =  salience_w1.abs().sum(dim=1)
        importance_w2_component =  salience_w2.abs().sum(dim=0)

        # analogous to group reduction?
        importance = importance_w1_component + importance_w2_component
        importances.append(importance.detach().cpu())
    return importances

def get_input(storage, key):
    """
    Get input into mlp layer in forward pass
    """
    def hook(module, input, output):
        # Assuming the layer takes a single Tensor as input, store it
        storage[key] = input[0].detach()
    return hook


def get_sample_input_trailing_token(mlp_input_dict):
    """
    Get single embedding(inputs into mlps) for each layer
    for trailing token in sequence
    """

    # alwyas take first sequence, is randomly sampled beforehand, so no need to random sample again
    # importances are based on the entire batch, but we only take one sample input
    return [inputs[0, -1, :].cpu() for layer, inputs in mlp_input_dict.items()]

def compute_delta_loss_importances(model, examples, idxs=None):
    """Computes and returns impotances of every hidden neuron in the model's
    mlps. Here we define importance as the change of the loss if we were to
    set the inbound and outbound weights of a neuron to 0."""
    
    mlps = get_mlps(model)
    mlps = mlps if idxs is None else [mlps[i] for i in idxs]

    mlp_input_dict = {}
    hooks = []
    
    for i, mlp in enumerate(mlps):
        fc1 = mlp.fc1
        hook = fc1.register_forward_hook(get_input(mlp_input_dict, i))
        hooks.append(hook)
        
    # compute and store first derivative squared
    loss = compute_acc_grad(model.cuda(), examples.cuda(), mlps)
    torch.cuda.synchronize()
    
    # compute and store first derivative
    loss = loss.mean()
    loss.backward()
    
    # Once first derivative and second derivative squared are stored,
    # compute the importances.
    importances = compute_mlp_importance(mlps)

    # cleanup
    for hook in hooks:
        hook.remove()

    sample_inputs_trailing_token = get_sample_input_trailing_token(mlp_input_dict)
    
    return importances, sample_inputs_trailing_token


def compute_random_importances(model, idxs=None):
    mlps = get_mlps(model)
    mlps = mlps if idxs is None else [mlps[i] for i in idxs]
    imps = {}
    for mlp in mlps:
        fc1 = mlp.fc1
        rng_imps = torch.randn(fc1.weight.shape[0])
        imps[mlp] = rng_imps
    return imps
