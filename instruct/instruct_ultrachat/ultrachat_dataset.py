import os
import json
from typing import *

import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

# from transformers.tokenization_utils import PreTrainedTokenizer
import copy

from torch.utils.data import get_worker_info


def load_single_file(data_file):
    with open(data_file)as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]


def load_raw_data(data_file):
    raw_dataset = []
    if isinstance(data_file, str):
        raw_dataset += load_single_file(data_file)
    elif isinstance(data_file, list):
        for f_ in data_file:
            raw_dataset += load_single_file(f_)
    return raw_dataset


# IGNORE_INDEX = -100  # huggingface tokenizer
IGNORE_INDEX = -1


def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 # tokenizer: PreTrainedTokenizer = None,
                 tokenizer=None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                 # split func for ddp
                 rank_id=0,
                 world_size=1,
                 ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"

        # ---------------- split func for ddp ---------------
        # worker_info = get_worker_info()
        # num_workers = worker_info.num_workers if worker_info is not None else 1
        # worker_id = worker_info.id if worker_info is not None else 0
        # num_shards = num_workers * self._num_processes
        # shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(raw_dataset) // world_size * world_size
        raw_dataset = raw_dataset[rank_id: max_num_files: world_size]
        # ----------------------------------------------------

        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, print("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", print("only tail truncate support")

    @property
    def end_token(self):
        if self._end_token is not None:
            return self._end_token
        end_token = self.sep[0]
        if end_token == "EOS":
            # self._end_token = self.tokenizer.eos_token  # for huggingface tokenizer
            self._end_token = self.tokenizer.processor.IdToPiece(self.tokenizer.eos_id)
        else:
            self._end_token = end_token
        return self._end_token

    def tokenize_example(self, example):
        # print(example)
        end_token = self.end_token
        tags = [i for _ in range(len(example["data"]) // 2) for i in ["User", "Assistant"]]
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            c_new = tags[i] + ": " + c + end_token
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                # tokenized = self.tokenizer(c_input, add_special_tokens=False)  # for huggingface transformer
                tokenized = self.tokenizer.encode(c_input, bos=False, eos=False, pad=False)

                tokenized_ids += tokenized  # ["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized)  # ["input_ids"])

                c_generate = c + end_token
                # tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized = self.tokenizer.encode(c_generate, bos=False, eos=False, pad=False)

                tokenized_ids += tokenized  # ["input_ids"]
                labels += tokenized  # ["input_ids"]

            else:
                # user
                if i == 0:
                    # no start token
                    # c_new = self.tokenizer.bos_token + tags[i] + ": " + c + end_token
                    c_new = self.tokenizer.processor.IdToPiece(self.tokenizer.bos_id) + tags[i] + ": " + c + end_token

                else:
                    c_new = self.start_token + tags[i] + ": " + c + end_token
                # tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized = self.tokenizer.encode(c_new, bos=False, eos=False, pad=False)
                tokenized_ids += tokenized  # ["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized)  # ["input_ids"])

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}
        # return torch.LongTensor(tokenized_ids), torch.LongTensor(labels)

    def truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]

        return tokenized_example

    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)

    @staticmethod
    def collate_fn(batch):
        # input_ids, labels = zip(*batch)  # transposed
        input_ids = [b['input_ids'] for b in batch]
        labels = [b['labels'] for b in batch]

        max_len = max(len(s) for s in input_ids)
        def pad_right(x, pad_id):
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
        x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
        y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
        return x, y


def load_struct_dataset():
    from pathlib import Path
    from tokenizer_team.llama_tokenizer import Tokenizer
    tokenizer_path = Path('/data/llm_models/llama_tokenizer/tokenizer.model')
    tokenizer = Tokenizer(tokenizer_path)

    raw_path = ['/data/data/ultrachat_data/ultrachat_material_release_230412.json']
    raw_dataset = load_raw_data(raw_path)
    dataset = PromptIterableDataset(raw_dataset, tokenizer=tokenizer, max_seq_length=2048, teacher_forcing=True)

    return dataset


if __name__ == "__main__":
    # from transformers import AutoTokenizer, LlamaTokenizer

    # TEMPLATE = "{} Assistant:"
    # tokenizer = LlamaTokenizer.from_pretrained("../../llama-7B-HF")
    from pathlib import Path
    from tokenizer_team.llama_tokenizer import Tokenizer

    tokenizer_path = Path('/data/llm_models/llama_tokenizer/tokenizer.model')
    tokenizer = Tokenizer(tokenizer_path)

    raw_path = ['/data/data/ultrachat_data/ultrachat_material_release_230412.json']
    raw_dataset = load_raw_data(raw_path)

    dataset = PromptIterableDataset(raw_dataset, tokenizer=tokenizer, max_seq_length=2048, teacher_forcing=True)
    print(f'dataset_len: {len(dataset)}')

    for data in dataset:
        print(data)
        print(tokenizer.decode(data["input_ids"][:1000]))

        model_output = data["input_ids"][:1000][data["labels"][:1000] != -100]
        print("##### model output")
        print(tokenizer.decode(model_output))
        break
