import os
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
import torch
from trl import SFTTrainer

random_seed=123
def to_dataset(iterable_dataset):
    data_list = [item for item in iterable_dataset]
    dataset = Dataset.from_dict({key: [dic[key] for dic in data_list] for key in data_list[0]})
    return dataset

def get_minipile(n=52000, do_split=True):
    dataset = load_dataset("JeanKaddour/minipile", streaming=True)
    dataset = to_dataset(dataset["train"].take(n))
    if do_split:
        dataset = dataset.to_pandas()
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=123)
    return dataset

def get_c4(n=52000, do_split=True):
    dataset = load_dataset("c4", "en", streaming=True)
    dataset = to_dataset(dataset["train"].take(n))
    if do_split:
        dataset = dataset.to_pandas()
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=123)
    return dataset

def get_wikitext2_filtered(n=52000, do_split=True):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = to_dataset(dataset["train"].filter(lambda example: len(example["text"]) >= 200).shuffle(seed=random_seed).select(range(0, n)))
    if do_split:
        dataset = dataset.to_pandas()
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=123)
    return dataset

def get_bookcorpus(n=52000, do_split=True):
    dataset = load_dataset("bookcorpus")
    dataset = to_dataset(dataset["train"].shuffle(seed=random_seed).select(range(0, n)))
    if do_split:
        dataset = dataset.to_pandas()
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=123)
    return dataset


def get_alpaca(tokenizer, n=52000, do_split=True):
    alpaca = load_dataset("tatsu-lab/alpaca")
    alpaca = alpaca["train"].shuffle(seed=random_seed).select(range(0, n))
    alpaca_template = {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"    
    }
    alpaca_list_ds = []
    for sample in alpaca.to_list():
        if sample["input"] != "":
            prompt = alpaca_template["prompt_input"].format(instruction=sample["instruction"], input=sample["input"])
        else:
            prompt = alpaca_template["prompt_no_input"].format(instruction=sample["instruction"])
        response = sample["output"]
        alpaca_list_ds.append((prompt, response))
    if do_split:
        train = TokenizedQADataset(alpaca_list_ds[:-2000], tokenizer)
        test = TokenizedQADataset(alpaca_list_ds[-2000:], tokenizer)
        return train, test
    return TokenizedQADataset(alpaca_list_ds, tokenizer)
    # dataset = TokenizedQADataset(alpaca_list_ds, tokenizer, max_length=256)


# class TokenizedDataset(torch.utils.data.Dataset):
#     def __init__(self, list_of_strings, tokenizer, max_length=2048):
#         self.data = []
#         self.tokenizer = tokenizer
#         self.total_calls = 0
#         self.total_length = 0
#         tokenizer.padding_side = "right"
#         pad = "do_not_pad"
#         self.max_length = max_length
#         for s in list_of_strings:
#             encoded = tokenizer(
#                 text=s + tokenizer.eos_token,
#                 return_tensors="np",
#                 truncation=True,
#                 max_length=self.max_length,
#                 padding=pad,
#             )
#             self.total_length += encoded['input_ids'].shape[1]
#             self.data.append({
#                 'input_ids': encoded['input_ids'].squeeze(0),
#                 'labels': encoded['input_ids'].squeeze(0),
#                 'attention_mask': encoded['attention_mask'].squeeze(0)
#             })
#         self.mean_length = self.total_length / len(list_of_strings)
#         self.packed_data = self.data.copy()
#         #self.pack(64)

#     def __len__(self):
#         return len(self.packed_data)

#     def __getitem__(self, idx):
#         return self.packed_data[idx]

# class TokenizedQADataset(TokenizedDataset):
#     """
#     Same as the tokenized dataset, but designed to "mask out" the labels of
#     the prompts such that they don't affect the model's loss.
    
#     Question and answer pairs are concatenated as they are given. Make sure
#     to include the right separator between them (" ", "\n", ". ", etc.)
#     """
#     def __init__(self, list_of_question_answers, tokenizer, max_length=2048):
#         self.data = []
#         self.tokenizer = tokenizer
#         self.total_calls = 0
#         self.total_length = 0
#         tokenizer.padding_side = "right"
#         self.max_length = max_length
#         for question, answer in list_of_question_answers:
#             encoded_question = tokenizer(
#                 text=question,
#                 return_tensors="np",
#                 truncation=True,
#                 max_length=self.max_length,
#                 padding="do_not_pad",
#             )
#             encoded_answer = tokenizer(
#                 text=answer+tokenizer.eos_token,
#                 return_tensors="np",
#                 truncation=True,
#                 max_length=self.max_length,
#                 padding="max_length",
#             )
#             encoded_inputs = np.concatenate([encoded_question["input_ids"],
#                                             encoded_answer["input_ids"]], axis=-1)
#             # labels have to be -100 so that the question does not affect the model's loss
#             encoded_labels = np.concatenate([np.array([-100] * encoded_question["input_ids"].shape[-1])[None, :],
#                                             encoded_answer["input_ids"]], axis=-1)
#             encoded_attention_mask = np.concatenate([encoded_question["attention_mask"],
#                                             encoded_answer["attention_mask"]], axis=-1)
#             encoded_labels[encoded_labels == tokenizer.eos_token_id] = -100

#             self.total_length += encoded_inputs.shape[1]
#             self.data.append({
#                 'input_ids': encoded_inputs.squeeze(0)[:max_length],
#                 'labels': encoded_labels.squeeze(0)[:max_length],
#                 'attention_mask': encoded_attention_mask.squeeze(0)[:max_length]
#             })
#         self.mean_length = self.total_length / len(list_of_question_answers)
#         self.packed_data = self.data.copy()

class SFTTrainer_(SFTTrainer):
    def _prepare_dataset(
        self,
        dataset,
        *args,
        **kwargs
    ):
        if isinstance(dataset, dict):
            return {k: self._prepare_dataset(v, *args, **kwargs) for k, v in dataset.items()}
        else:
            return super()._prepare_dataset(dataset, *args, **kwargs)


from loguru import logger
import numpy as np
import random
from transformers.data.data_collator import DataCollatorMixin, DataCollatorForLanguageModeling
import torch


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_strings, tokenizer, max_length=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.total_calls = 0
        self.total_length = 0
        tokenizer.padding_side = "right"
        pad = "do_not_pad"
        self.max_length = max_length
        for s in list_of_strings:
            encoded = tokenizer(
                text=s + tokenizer.eos_token,
                return_tensors="np",
                truncation=True,
                max_length=self.max_length,
                padding=pad,
            )
            self.total_length += encoded['input_ids'].shape[1]
            self.data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'labels': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        self.mean_length = self.total_length / len(list_of_strings)
        logger.info(f"Mean length of tokens per window: {self.mean_length}")
        self.packed_data = self.data.copy()
        #self.pack(64)


    def pack(self, N):
        data_pack = self.data.copy()
        self.packed_data = []
        total_length = 0
        while data_pack:
            combined_item = {'input_ids': np.array([], dtype=np.int64), 'labels': np.array([], dtype=np.int64),
                             'attention_mask': np.array([], dtype=np.int64)}
            current_length = 0
            items_to_remove = []
            sample_size = min(N, len(data_pack))
            sampled_indices = random.sample(range(len(data_pack)), sample_size)
            random.shuffle(sampled_indices)  # Shuffle to ensure random pick order

            for idx in sampled_indices:
                item = data_pack[idx]
                item_length = len(item['input_ids'])
                if current_length + item_length <= self.max_length:
                    for key in combined_item:
                        combined_item[key] = np.concatenate([combined_item[key], item[key]]) if combined_item[key].size else item[key]
                    current_length += item_length
                    items_to_remove.append(idx)

            total_length += current_length
            # Padding to reach max_length - not sure why, but it seems to improve accuracy
            padding_length = self.max_length - current_length
            if padding_length > 0:
                pad_token_id = self.tokenizer.eos_token_id
                if padding_length > 0:
                    combined_item["labels"] = np.pad(combined_item["labels"], (0, padding_length), constant_values=-100)
                    combined_item["input_ids"] = np.pad(combined_item["input_ids"], (0, padding_length), constant_values=pad_token_id)
                    combined_item["attention_mask"] = np.pad(combined_item["attention_mask"], (0, padding_length), constant_values=0)

            # Ensure we always have at least one item to add to avoid empty data
            if items_to_remove:
                self.packed_data.append(combined_item)
                # Remove items from self.data in reverse order to avoid index issues
                for idx in sorted(items_to_remove, reverse=True):
                    del data_pack[idx]
            else:
                # Break the loop if no items were suitable to avoid infinite loop
                logger.warning("No items were suitable for packing, breaking loop")
                break
        logger.info(f"Mean length of tokens per packed window: {total_length / len(self.packed_data)}")
        self.total_calls = 0

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        #self.total_calls += 1
        #if self.total_calls > len(self.packed_data):
        #    prev_len = len(self.packed_data)
        #    self.pack(64)
        #    while len(self.packed_data) < prev_len:
        #        idx = random.randint(0, len(self.packed_data) - 1)
        #        self.packed_data.append(self.packed_data[idx])
        return self.packed_data[idx]

class TokenizedQADataset(TokenizedDataset):
    """
    Same as the tokenized dataset, but designed to "mask out" the labels of
    the prompts such that they don't affect the model's loss.
    
    Question and answer pairs are concatenated as they are given. Make sure
    to include the right separator between them (" ", "\n", ". ", etc.)
    """
    def __init__(self, list_of_question_answers, tokenizer, max_length=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.total_calls = 0
        self.total_length = 0
        tokenizer.padding_side = "right"
        pad = "do_not_pad"
        self.max_length = max_length
        for question, answer in list_of_question_answers:
            encoded_question = tokenizer(
                text=question,
                return_tensors="np",
                truncation=True,
                max_length=self.max_length,
                padding=pad,
            )
            encoded_answer = tokenizer(
                text=answer+tokenizer.eos_token,
                return_tensors="np",
                truncation=True,
                max_length=self.max_length,
                padding=pad,
            )
            encoded_inputs = np.concatenate([encoded_question["input_ids"],
                                            encoded_answer["input_ids"]], axis=-1)
            # labels have to be -100 so that the question does not affect the model's loss
            encoded_labels = np.concatenate([np.array([-100] * encoded_question["input_ids"].shape[-1])[None, :],
                                            encoded_answer["input_ids"]], axis=-1)
            encoded_attention_mask = np.concatenate([encoded_question["attention_mask"],
                                            encoded_answer["attention_mask"]], axis=-1)
            self.total_length += encoded_inputs.shape[1]
            self.data.append({
                'input_ids': encoded_inputs.squeeze(0),
                'labels': encoded_labels.squeeze(0),
                'attention_mask': encoded_attention_mask.squeeze(0)
            })
        self.mean_length = self.total_length / len(list_of_question_answers)
        logger.info(f"Mean length of tokens per window: {self.mean_length}")
        self.packed_data = self.data.copy()

#        self.pack(64)

class QADataCollator(DataCollatorMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.return_tensors = "pt"
        self.lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def fill(self, tens, mx_len, val):
        extras = mx_len - len(tens)
        extras = torch.ones(extras, dtype=tens.dtype) * val
        return torch.concat([tens, extras])
    
    def torch_call(self, batch):
        if "labels" not in batch[0]:
            return self.lm_data_collator.torch_call(batch)
        inp_ids = [torch.from_numpy(b["input_ids"]) for b in batch]
        labels = [torch.from_numpy(b["labels"]) for b in batch]
        mask = [torch.from_numpy(b["attention_mask"]) for b in batch]
        mx_len = max(map(len, inp_ids))
        
        inp_ids = [self.fill(x, mx_len, self.tokenizer.eos_token_id) for x in inp_ids]
        labels = [self.fill(x, mx_len, -100) for x in labels]
        mask = [self.fill(x, mx_len, 1) for x in mask]
        batch = {
            "input_ids":torch.stack(inp_ids),
            "labels":torch.stack(labels),
            "attention_mask":torch.stack(mask)
        }
        return batch

