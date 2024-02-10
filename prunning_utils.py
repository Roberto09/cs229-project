def get_mlps(model):
    layers = model.get_submodule("model").get_submodule("layers")
    return [layer.get_submodule("mlp") for layer in layers]