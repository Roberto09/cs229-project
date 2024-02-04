from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import pickle

def fetch_data(data="nampdn-ai/tiny-textbooks"):
    dataset = load_dataset(data)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

    num_rows = dataset['train'].num_rows
    dataset_tokenized = [tokenizer.encode(dataset['train'][i]['text']) for i in range(num_rows)]

    token_row_map = defaultdict(set)

    for i, row in enumerate(dataset_tokenized):
        for token in row:
            token_row_map[token].add(i)
    
    with open("dataset_tokenized.pkl", "wb") as f:
        pickle.dump(dataset_tokenized, f)

    with open("token_row_map.pkl", "wb") as f:
        pickle.dump(token_row_map, f)
    