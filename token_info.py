import random
import pickle
from collections import Counter
from itertools import chain

class TokenInfo():

    # TODO: maybe take dataset name as argument, difficult because they dont all have the same structure
    def __init__(self):        
        print('...Loading dataset...')
        with open("dataset_tokenized.pkl", "rb") as f:
            self.dataset_tokenized = pickle.load(f)
        
        with open("token_row_map.pkl", "rb") as f:
            self.token_row_map = pickle.load(f)

        self.token_counts = Counter(list(chain(*self.dataset_tokenized)))
  
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
        return self.token_counts.most_common(n)


if __name__ == '__main__':
    token_info = TokenInfo()
    prefix_test = token_info.get_prefixes('dog', 10, 5)
    top_tokens = token_info.top_n(100)

    print(prefix_test)
