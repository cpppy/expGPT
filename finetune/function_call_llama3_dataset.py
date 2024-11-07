# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

# import numpy as np
# np.random.rand()


class FunctionCallingDataset(Dataset):
    def __init__(self, data_path, tokenizer, partition="train", max_words=512):
        samples = []
        with open(data_path, 'r') as f:
            json_list = list(f)
            for json_str in json_list:
                sample = json.loads(json_str)
                samples.append(sample)
        # if 'train_filtered' in data_path:
        #     # self.samples = samples[0:10000]
        #     self.samples = samples[10000:20000]
        # else:
        #     self.samples = samples[0:1000]
        self.samples = samples
        print(f'n_sample: {len(samples)}')

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt_text = sample['text'].split('<functioncall>')[0]
        example_text = sample['text']
        prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt = torch.tensor(prompt, dtype=torch.int64)
        example = self.tokenizer.encode(example_text, add_special_tokens=False)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }


def main():

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/data/data/bloomz-560m')

    dataset = FunctionCallingDataset(data_path='/data/data/agent_tuning_data/function-calling-llama-3-format-v1.1/valid_filtered.jsonl',
                                     tokenizer=tokenizer)
    s = dataset.__getitem__(9)
    print(s)


if __name__=='__main__':

    main()

