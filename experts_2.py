from __future__ import annotations

import re
from itertools import chain
from torch import nn

import torch
from torch import nn
from tqdm import tqdm
import inspect
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import dataclasses
from cluster_router import ClusterRouter

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import get_quantization_config

from peft.tuners.lora.layer import Conv2d, Embedding, Linear, LoraLayer
from peft import LoraModel
from transformers.activations import ACT2FN
from topk_perceptron_router import TopKPerceptronRouter

def get_lora_adapter(model, target, lora_config, current_key, adapter_name="default"):
    optional_kwargs = {
        "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
        "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
        "current_key": current_key,
    }
    # almost identical copy of LoraModel._create_and_replace
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")
    # Regexp matching - Find key which matches current target_name in patterns provided
    pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
    target_name_key = next(filter(lambda key: re.match(f".*\.{key}$", current_key), pattern_keys), current_key)

    r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
    alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
    bias = hasattr(target, "bias") and target.bias is not None
    kwargs = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": lora_config.lora_dropout,
        "fan_in_fan_out": lora_config.fan_in_fan_out,
        "init_lora_weights": lora_config.init_lora_weights,
    }
    # kwargs["loaded_in_8bit"] = optional_kwargs.pop("loaded_in_8bit", False)
    # kwargs["loaded_in_4bit"] = optional_kwargs.pop("loaded_in_4bit", False)
    kwargs["bias"] = bias

    quantization_config = get_quantization_config(model, method="gptq")
    if quantization_config is not None:
        kwargs["gptq_quantization_config"] = quantization_config

    linear_types = (Linear,)
    if is_bnb_available():
        from peft.tuners.lora.bnb import Linear8bitLt

        linear_types += (Linear8bitLt,)
    if is_bnb_4bit_available():
        from peft.tuners.lora.bnb import Linear4bit

        linear_types += (Linear4bit,)

    # TODO: better deal with that
    if isinstance(target, Conv2d):
        assert False, "are you sure you know what you are doing?"
        target.update_layer_conv2d(
            adapter_name,
            r,
            alpha,
            lora_config.lora_dropout,
            lora_config.init_lora_weights,
        )
    elif isinstance(target, Embedding):
        assert False, "are you sure you know what you are doing?"
        target.update_layer_embedding(
            adapter_name,
            r,
            alpha,
            lora_config.lora_dropout,
            lora_config.init_lora_weights,
        )
    elif isinstance(target, linear_types):
        assert False, "are you sure you know what you are doing?"
        target.update_layer(
            adapter_name,
            r,
            alpha,
            lora_config.lora_dropout,
            lora_config.init_lora_weights,
        )
    else:
        new_module = LoraModel._create_new_module(lora_config, adapter_name, target, **kwargs)
        # if adapter_name != self.active_adapter:
        #     # adding an additional adapter: it is not automatically trainable
        #     new_module.requires_grad_(False)
        _prepare_module(new_module, target)
        return new_module

def _prepare_module(
    new_module, # new module
    child, # old module
    ):
    # It's not necessary to set requires_grad here, as that is handled by
    # _mark_only_adapters_as_trainable

    # child layer wraps the original module, unpack it
    if hasattr(child, "base_layer"):
        child = child.base_layer

    if not hasattr(new_module, "base_layer"):
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

    if getattr(child, "state", None) is not None:
        if hasattr(new_module, "base_layer"):
            new_module.base_layer.state = child.state
        else:
            new_module.state = child.state
        new_module.to(child.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        prefix = "lora_"
        if (prefix in name) or ("ranknum" in name):
            weight = child.qweight if hasattr(child, "qweight") else child.weight
            module.to(weight.device)

def mark_adapters_and_routers_as_trainable(model, lora_config, adapter_name="default"):
    lora_prefix, router_prefix = "lora_", "_router"
    active_adapter = adapter_name
    for n, p in model.named_parameters():
        if not (lora_prefix in n or router_prefix in n):
            p.requires_grad = False

    bias = lora_config.bias
    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
    

class ModifiedLora(nn.Module):
    def __init__(self, orig_lora, activation_fn, is_fc1):
        super().__init__()
        self.orig_lora = orig_lora
        self.is_fc1 = is_fc1
        self.activation_fn = activation_fn
    
    def forward(self, x: torch.Tensor, x2=None, *args, **kwargs) -> torch.Tensor:
        # This is just copied from the original lora.Linear forward
        previous_dtype = x.dtype
        if self.orig_lora.disable_adapters:
            assert False
            if self.orig_lora.merged:
                self.orig_lora.unmerge()
            result = self.orig_lora.base_layer(x, *args, **kwargs)
        elif self.orig_lora.merged:
            assert False
            result = self.orig_lora.base_layer(x, *args, **kwargs)
        else:
            assert len(self.orig_lora.active_adapters) == 1
            active_adapter = next(iter(self.orig_lora.active_adapters))
            if active_adapter not in self.orig_lora.lora_A.keys():
                assert False
            lora_A = self.orig_lora.lora_A[active_adapter]
            lora_B = self.orig_lora.lora_B[active_adapter]
            dropout = self.orig_lora.lora_dropout[active_adapter]
            scaling = self.orig_lora.scaling[active_adapter]
            h1 = self.orig_lora.base_layer(x, *args, **kwargs)
            if self.is_fc1:
                h2 = lora_B(lora_A(dropout(x))) * scaling
                result = self.activation_fn(h1).to(previous_dtype), self.activation_fn(h2).to(previous_dtype)
            else:
                h2 = lora_B(lora_A(dropout(x2))) * scaling
                result = self.activation_fn(h1 + h2).to(previous_dtype)
        return result

class Experts(nn.Module):
    def __init__(
        self,
        model,
        phi_mlp,
        lora_config,
        layer,
        curr_token_idx_tracker,
        phi_mlp_config,
        num_experts=8,
        K=None,
        use_improved_lora=False,
    ):
        super().__init__()
        self.K = K
        self.num_experts = num_experts
        self.config = phi_mlp_config
        self.use_improved_lora = use_improved_lora
        self.activation_fn = ACT2FN[self.config.hidden_act]
        self.curr_token_idx = curr_token_idx_tracker
        self.router = self._get_init_router(num_experts, layer, K, curr_token_idx_tracker)
        
        experts = [self.get_expert(model, phi_mlp, lora_config) for i in range(num_experts)]
        self.experts_fc1 = nn.ModuleList([exp[0] for exp in experts])
        self.experts_fc2 = nn.ModuleList([exp[1] for exp in experts])

    def _get_init_router(self, num_experts, layer, K, curr_token_idx_tracker):
        # Use cluster router
        if K is None:
            assert curr_token_idx_tracker is not None
            return ClusterRouter(layer, 52_000, num_experts)
        # Use mlp router
        return TopKPerceptronRouter(2048, num_experts, layer, K) # hardcode input size

    def get_expert(self, model, phi_mlp, lora_config):
        # TODO: Does this even makes sense? could this cause multiple copies of lora adapters?
        lora_fc1 = get_lora_adapter(model, phi_mlp.fc1, lora_config, "fc1")
        lora_fc2 = get_lora_adapter(model, phi_mlp.fc2, lora_config, "fc2")
        if self.use_improved_lora:
            lora_fc1 = ModifiedLora(lora_fc1, self.activation_fn, is_fc1=True)
            lora_fc2 = ModifiedLora(lora_fc2, self.activation_fn, is_fc1=False)
        return lora_fc1, lora_fc2

    def groupby_experts(self, experts, embeds):
        orig_idxs_tens = torch.arange(len(embeds), device = 'cuda:0')
        embeds_per_expert = []
        embeds_orig_idxs = []
        for e in range(self.num_experts):
            mask = experts == e
            curr_expert_embeds = embeds[mask]
            embeds_per_expert.append(curr_expert_embeds)
            embeds_orig_idxs.append(orig_idxs_tens[mask])
        return embeds_per_expert, embeds_orig_idxs

    def forward_expert(self, embs, exp_fc1, exp_fc2):
        if self.use_improved_lora:
            # activation function is done insisde the experts
            h1, h2 = exp_fc1(embs)
            return exp_fc2(x=h1, x2=h2)
        else:
            return exp_fc2(self.activation_fn(exp_fc1(embs)))

    def forward_cluter_router(self, hidden_states):
        # token ids?
        expert_idxs = self.router(self.curr_token_idx)
        # TODO: change the :hiden_states ... stuff to self.curr_token_idx instead
        expert_idxs = expert_idxs[:hidden_states.shape[0], :hidden_states.shape[1]] # (batch_size, tokens)
        experts = expert_idxs.flatten()
        
        orig_embeds_shape = hidden_states.shape
        embeds = hidden_states.reshape(orig_embeds_shape[0]*orig_embeds_shape[1], -1)
        embeds_per_expert, embeds_orig_idxs = self.groupby_experts(experts, embeds)

        res = torch.zeros(embeds.shape, dtype=embeds.dtype, device=embeds.device)
        for embs, emb_idxs, exp_fc1, exp_fc2\
            in zip(embeds_per_expert, embeds_orig_idxs, self.experts_fc1, self.experts_fc2):
            embs = self.forward_expert(embs, exp_fc1, exp_fc2)
            res[emb_idxs] = embs
        return res.reshape(orig_embeds_shape)

    def forward_mlp_router(self, hidden_states):
        batch_size, seq_len, feature_dim = hidden_states.shape
        expert_idxs, expert_weights = self.topk_router(hidden_states)  # Shape: [batch_size, seq_len, k], [batch_size, seq_len, k]

        res = torch.zeros_like(hidden_states)

        # for each embedding, process one expert at a time(could potentially optimize later)
        # group together embeddings that share expert i as their kth expert
        for k in range(self.K):
            experts = expert_idxs[:, :, k].flatten() # take k-th expert
            weights = expert_weights[:, :, k].flatten()
            embeds = hidden_states.view(-1, feature_dim)  # Flatten to [batch_size*seq_len, feature_dim]

            embeds_per_expert, embeds_orig_idxs = self.groupby_experts(experts, embeds)

            for embs, emb_idxs, exp_fc1, exp_fc2 in zip(embeds_per_expert, embeds_orig_idxs, self.experts_fc1, self.experts_fc2):
                # Process the embeddings for this expert
                processed_embs = self.forward_expert(embs, exp_fc1, exp_fc2)
                # Use weights to scale the processed embeddings
                weighted_embs = processed_embs * weights[emb_idxs].unsqueeze(1)
                # Accumulate results
                res.view(-1, feature_dim)[emb_idxs] += weighted_embs
        return res
    
    def forward(self, hidden_states):
        # Cluster Router
        if self.K is None:
            return self.forward_cluter_router(hidden_states)
        # MLP router
        return self.forward_mlp_router(hidden_states)

class EmbeddingTokenIdxTracker(nn.Module):
    def __init__(self, orig_embed_layer):
        super().__init__() 
        self.idx_tracker = torch.zeros((2048, 2048), dtype=int)
        self.embed = orig_embed_layer

    def forward(self, inp_ids):
        self.idx_tracker[:inp_ids.shape[0], :inp_ids.shape[1]] = inp_ids
        return self.embed(inp_ids)
    


def prepare_as_if_peft_model(model, training_arguments, config):
    args = training_arguments
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        _support_gc_kwargs = hasattr(
            args, "gradient_checkpointing_kwargs"
        ) and "gradient_checkpointing_kwargs" in list(
            inspect.signature(prepare_model_for_kbit_training).parameters
        )

        preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

        if _support_gc_kwargs:
            preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

        model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

        args = dataclasses.replace(args, gradient_checkpointing=False)
    return args

def prepare_model_for_gradient_checkpointing(model):
    r"""
    Prepares the model for gradient checkpointing if necessary
    """
    if not (
        getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_quantized", False)
    ):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model