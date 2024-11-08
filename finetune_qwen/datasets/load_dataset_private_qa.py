from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
# from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from tqdm import tqdm

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id

    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


class LazySupervisedDatasetMed(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDatasetMed, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


# def make_supervised_data_module(
#     tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
# ) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     dataset_cls = (
#         LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
#         # LazySupervisedDatasetMed
#     )
#     rank0_print("Loading data...")
#
#     train_json = json.load(open(data_args.data_path, "r"))
#     train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)
#
#     if data_args.eval_data_path:
#         eval_json = json.load(open(data_args.eval_data_path, "r"))
#         eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
#     else:
#         eval_dataset = None
#
#     return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# def load_texts(data_file, expected_size=None):
#     texts = []
#     for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
#         texts.append(json.loads(line))
#     return texts


import jsonlines
def load_texts(data_file, expected_size=None):
    texts = []
    for line in tqdm(jsonlines.open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line))
    return texts


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        # LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
        LazySupervisedDatasetMed
    )
    rank0_print("Loading data...")

    if len(data_args.data_path) > 0:
        samples = []
        for data_path in data_args.data_path:
            print(f'data_path: {data_path}')
            raw_datas = load_texts(data_path)  # [0:10]
            for data in raw_datas:
                try:
                    data = {k.strip(): v.strip() for k, v in data.items()}
                    messages = []
                    messages.append({"from": "user", "value": data['instruction'] + data['input']})
                    messages.append({"from": "assistant", "value": data['output']
                                     })
                    # print(messages)
                    samples.append(dict(conversations=messages))
                except Exception as e:
                    # print(f'[WARNING] Fail to parse sample: {data}')
                    # print(e)
                    continue

        train_dataset = dataset_cls(samples, tokenizer=tokenizer, max_len=max_len)
    else:
        raise ValueError

    if len(data_args.eval_data_path) > 0:
        samples = []
        for data_path in data_args.eval_data_path:
            print(f'data_path: {data_path}')
            raw_datas = load_texts(data_path)  # [0:10]
            for data in raw_datas:
                messages = []
                messages.append({"from": "user", "value": data['instruction'] + data['input']})
                messages.append({"from": "assistant", "value": data['output']
                                 })
                # print(messages)
                samples.append(dict(conversations=messages))

        eval_dataset = dataset_cls(samples, tokenizer=tokenizer, max_len=max_len)

    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


@dataclass
class DataArguments:
    data_path = [
        # '/mnt2/expGPT/data/qa_samples_from_baiduyidian_20241010_finish.jsonl'
        '../data/qa2_samples_teacher_and_student_from_baiduyidian_20241023_finish_inst.jsonl'
    ]
    eval_data_path = [
        # '/data/data/med_dataset/medical/finetune/valid_zh_0.json',
        # '/data/data/med_dataset/medical/finetune/valid_en_1.json',
    ]
    lazy_preprocess = True


def load_dataset(tokenizer):
    data_args = DataArguments()

    # tokenizer.pad_token_id = tokenizer.eod_id
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.im_start_token = '<|im_start|>'
    tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    # tokenizer.im_end_token = '<|im_end|>'
    tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=tokenizer.model_max_length
    )

    print(f'n_train: {len(data_module["train_dataset"])}')
    print(f'n_val: {len(data_module["eval_dataset"]) if data_module["eval_dataset"] else 0}')
    return data_module


if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/data/data/Qwen2-0.5B-Instruct",
        # cache_dir=training_args.cache_dir,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    load_dataset(tokenizer)
