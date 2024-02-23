from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import pickle
import os
import pandas as pd

def fetch_preprocessed_data(data="nampdn-ai/tiny-textbooks", filedir="./"):
    """ Gets data from huggingface and:
    * Tokenizes the whole dataset and pickles it to filedir/dataset_tokenized.pkl
    * Creates a map of token to sample and pickles it to filedir/token_row_map.pkl
    
    If filedir/{dataset_tokenized.pkl, filedir/token_row_map.pkl} already exist,
    it just reads the values and returns them
    """

    tokenized_dataset_path = f"{filedir}/dataset_tokenized.pkl"
    token_row_map_path = f"{filedir}/token_row_map.pkl"
    if os.path.isfile(tokenized_dataset_path) and os.path.isfile(token_row_map_path):
        return pd.read_pickle(tokenized_dataset_path), pd.read_pickle(token_row_map_path)

    dataset = load_dataset(data)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

    num_rows = dataset['train'].num_rows
    dataset_tokenized = [tokenizer.encode(dataset['train'][i]['text']) for i in range(num_rows)]

    token_row_map = defaultdict(set)

    for i, row in enumerate(dataset_tokenized):
        for token in row:
            token_row_map[token].add(i)
    
    pd.to_pickle(dataset_tokenized, tokenized_dataset_path)
    pd.to_pickle(token_row_map, token_row_map_path)
    return dataset_tokenized, token_row_map_path

import random
import pickle
from collections import Counter
from transformers import AutoTokenizer
from itertools import chain

class TokenInfo():

    # TODO: maybe take dataset name as argument, difficult because they dont all have the same structure
    def __init__(self, filedir="./"):        
        print('...Loading dataset...')
        with open(f"{filedir}/dataset_tokenized.pkl", "rb") as f:
            self.dataset_tokenized = pickle.load(f)
        
        with open(f"{filedir}/token_row_map.pkl", "rb") as f:
            self.token_row_map = pickle.load(f)

        self.token_counts = Counter(list(chain(*self.dataset_tokenized)))
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
  
        print('...Loading complete...')
    
    def get_prefixes(self, token, prefix_len, n):
        token_rows = list(self.token_row_map[token])
        out = []
        while len(out) < n:
            row = random.choice(token_rows)
            row_tokens = self.dataset_tokenized[row]
            token_idx = [index for index, value in enumerate(row_tokens) if value == token and index >= prefix_len]
            if len(token_idx) > 0:
                i = random.sample(token_idx, 1)[0]
                out.append(row_tokens[i-prefix_len: i+1])

        return out

    def top_n(self, n):
        top_tokens =  self.token_counts.most_common(n)
        return [(x[0], self.tokenizer.decode(x[0]), x[1]) for x in top_tokens]


if __name__ == '__main__':
    # EXAMPLE USAGE
    token_info = TokenInfo()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    prefix_test = token_info.get_prefixes(tokenizer.encode('dog')[0], 10, 5)
    top_tokens = token_info.top_n(100)

    print(prefix_test)
