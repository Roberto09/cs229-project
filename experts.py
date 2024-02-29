from __future__ import annotations

import re
from itertools import chain

import torch
from torch import nn
from tqdm import tqdm
import inspect
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import dataclasses

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import get_quantization_config

from peft.tuners.lora.layer import Conv2d, Embedding, Linear, LoraLayer
from peft import LoraModel
from transformers.activations import ACT2FN

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

def mark_only_adapters_as_trainable(model, lora_config, adapter_name="default"):
    prefix = "lora_"
    active_adapter = adapter_name
    for n, p in model.named_parameters():
        if prefix not in n:
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
    
from cluster_router import ClusterRouter

class Experts(nn.Module):
    def __init__(self, model, phi_mlp, lora_config, layer, curr_token_idx_tracker, phi_mlp_config, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.config = phi_mlp_config
        self.activation_fn = ACT2FN[self.config.hidden_act]

        self.cluster_router = ClusterRouter(layer, 52_000, num_experts)
        experts = [self.get_expert(model, phi_mlp, lora_config) for i in range(num_experts)]
        self.experts_fc1 = nn.ModuleList([exp[0] for exp in experts])
        self.experts_fc2 = nn.ModuleList([exp[1] for exp in experts])
        self.curr_token_idx = curr_token_idx_tracker

    def get_expert(self, model, phi_mlp, lora_config):
        # TODO: Does this even makes sense? could this cause multiple copies of lora adapters?
        lora_fc1 = get_lora_adapter(model, phi_mlp.fc1, lora_config, "fc1")
        lora_fc2 = get_lora_adapter(model, phi_mlp.fc2, lora_config, "fc2")
        return lora_fc1, lora_fc2

    def groupby_experts(self, experts, embeds):
        orig_idxs_tens = torch.arange(len(embeds))
        embeds_per_expert = []
        embeds_orig_idxs = []
        for e in range(self.num_experts):
            mask = experts == e
            curr_expert_embeds = embeds[mask]
            embeds_per_expert.append(curr_expert_embeds)
            embeds_orig_idxs.append(orig_idxs_tens[mask])
        return embeds_per_expert, embeds_orig_idxs

    def forward(self, hidden_states):
        # token ids?
        expert_idxs = self.cluster_router(self.curr_token_idx)
        expert_idxs = expert_idxs[:hidden_states.shape[0], :hidden_states.shape[1]] # (batch_size, tokens)
        experts = expert_idxs.flatten()
        
        orig_embeds_shape = hidden_states.shape
        embeds = hidden_states.reshape(orig_embeds_shape[0]*orig_embeds_shape[1], -1)
        embeds_per_expert, embeds_orig_idxs = self.groupby_experts(experts, embeds)

        res = torch.zeros(embeds.shape, dtype=embeds.dtype, device=embeds.device)
        for embs, emb_idxs, exp_fc1, exp_fc2\
            in zip(embeds_per_expert, embeds_orig_idxs, self.experts_fc1, self.experts_fc2):
            embs = exp_fc2(self.activation_fn(exp_fc1(embs)))
            res[emb_idxs] = embs
       
        return res.reshape(orig_embeds_shape)


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