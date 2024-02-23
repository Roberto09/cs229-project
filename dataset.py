import os
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd

def get_baseline_dataset(filename="./baseline_dataset.pkl"):
    """Returns tiny-textbooks dataset with 52 examples and caches it
    in filename.
    """
    if os.path.isfile(filename):
        print("reading pickle")
        return pd.read_pickle(filename)
    dataset = load_dataset("nampdn-ai/tiny-textbooks")
    
    np.random.seed(123)
    data_idxs = np.random.permutation(np.arange(len(dataset["train"])))[:52000]
    
    train_data = dataset["train"]
    train_data_pd = train_data.to_pandas()
    train_data_pd = train_data_pd.iloc[data_idxs]
    train_data_pd = train_data_pd.reset_index(drop=True)
    dataset = Dataset.from_pandas(train_data_pd)
    dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=123)
    pd.to_pickle(dataset, "./baseline_dataset.pkl")
    return dataset