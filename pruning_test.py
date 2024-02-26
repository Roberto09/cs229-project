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

"""def get_importances():
    print("getting less average importances, this is pretty bad, should fix!!!")
    dir = "./importances_data/importances"
    imp_files = os.listdir(dir)
    imp_files = [file for file in imp_files if file.endswith(".pkl")][:100]
    importances = {}
    for imp_file in tqdm(imp_files):
        importances.update(pd.read_pickle(f"{dir}/{imp_file}"))
    
    return importances

imps = get_importances()

def get_avg_imporances(importances):
    avg_imps = [torch.zeros_like(imp) for imp in list(importances.values())[0]]
    for token, imps in tqdm(importances.items()):
        for i, layer_imps in enumerate(imps):
            avg_imps[i] += layer_imps / len(importances)
    # TODO think harder about averaging method
    return avg_imps
"""

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
# from importances import get_mlps

prune_mlps_holistically(avg_imps, 0.2)