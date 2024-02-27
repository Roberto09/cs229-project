import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
import itertools
import pandas as pd
import os
from dataset_preprocessing import TokenInfo
import torch
from tqdm import tqdm

import os
from os import listdir

model_id = "microsoft/phi-1_5"
model_revision = "349cf8b5e81fd5f791d1740da5de1313a0419bbd" # latest as of feb 1st

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=model_revision,
    trust_remote_code=True,
    # be careful with this?
    # torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2",
)

def get_mlps(model):
    layers = model.get_submodule("model").get_submodule("layers")
    return [layer.get_submodule("mlp") for layer in layers]

mlps = get_mlps(model)

def get_lm_prunner_style_importances(model):
    mlps = get_mlps(model)
    imps = {}
    imps_list = pd.read_pickle("average_importances_sorvisto.pkl")
    for mlp, imp in zip(mlps, imps_list):
        imps[mlp] = imp
    return imps

avg_imps = get_lm_prunner_style_importances(model)

from prunners import prune_mlps_holistically

prune_mlps_holistically(avg_imps, 0.2)