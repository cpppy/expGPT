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
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from tqdm import tqdm
import pickle

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import jsonlines

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess_for_instruct(
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


def preprocess_for_pretrain(
        token_ids,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
) -> Dict:
    n_total = len(token_ids)
    n_chunks = (n_total - 1) // max_len

    input_ids, targets, masks = [], [], []
    for idx in range(n_chunks):
        _input_ids = token_ids[idx * max_len: (idx + 1) * max_len]
        _targets = token_ids[idx * max_len + 1: (idx + 1) * max_len + 1]
        input_ids.append(_input_ids)
        targets.append(_targets)
        masks.append(list(map(lambda x: x != tokenizer.pad_token_id, _input_ids)))
    # input_ids = torch.tensor(input_ids, dtype=torch.int)
    # targets = torch.tensor(targets, dtype=torch.int)
    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
        attention_mask=masks,
    )


class PretrainDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int,
                 cache=None):
        super(PretrainDataset, self).__init__()

        rank0_print("Formatting inputs...")

        ################# load raw_data ################
        if cache is not None and os.path.exists(cache):
            print(f'Load from cache: {cache}')
            with open(cache, 'rb') as f:
                data_cache = pickle.load(f)
        else:
            sources = []
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            for data_path in raw_data:
                with jsonlines.open(data_path, 'r') as f:
                    sources.extend([line['text']
                                    for line in tqdm(f, desc=os.path.basename(data_path))])
            print(f'n_texts: {len(sources)}')
            # sources = sources[0:100]

            eos_token = "<|endoftext|>"
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            all_token_ids = []
            for text in tqdm(sources, total=len(sources), desc='tokenizing'):
                _token_ids = tokenizer.encode(text)
                all_token_ids.extend(_token_ids + [eos_token_id])
            print(f'n_total_token_ids: {len(all_token_ids)}')

            data_cache = preprocess_for_pretrain(all_token_ids, tokenizer, max_len)

            # TODO: add rank check
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(data_cache, f)

        self.input_ids = data_cache["input_ids"]
        self.labels = data_cache["labels"]
        self.attention_mask = data_cache["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            # input_ids=self.input_ids[i],
            # labels=self.labels[i],
            # attention_mask=self.attention_mask[i],
            input_ids=torch.tensor(self.input_ids[i], dtype=torch.int),
            labels=torch.tensor(self.labels[i], dtype=torch.int),
            attention_mask=torch.tensor(self.attention_mask[i], dtype=torch.int),
        )


#
# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""
#
#     def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
#         super(LazySupervisedDataset, self).__init__()
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.raw_data = raw_data
#         self.cached_data_dict = {}
#
#     def __len__(self):
#         return len(self.raw_data)
#
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         if i in self.cached_data_dict:
#             return self.cached_data_dict[i]
#
#         ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
#         ret = dict(
#             input_ids=ret["input_ids"][0],
#             labels=ret["labels"][0],
#             attention_mask=ret["attention_mask"][0],
#         )
#         self.cached_data_dict[i] = ret
#
#         return ret


def load_texts(data_file, expected_size=None):
    texts = []
    for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line))
    return texts

#
# def make_supervised_data_module(
#         tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
# ) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     dataset_cls = (
#         # LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
#         # LazySupervisedDatasetMed
#
#     )
#     rank0_print("Loading data...")
#
#     # train_json = json.load(open(data_args.data_path, "r"))
#     texts = load_texts(data_args.data_path)  # [0:1]
#     samples = []
#     for text in texts:
#         messages = []
#         for m in text['data']:
#             if '问：' in m:
#                 messages.append({"from": "user", "value": m.replace("问：", "")})
#             elif '答：' in m:
#                 messages.append({"from": "assistant", "value": m.replace("问：", "")})
#             else:
#                 logging.warning(f'Failed to parse raw text: {m}')
#                 raise ValueError
#
#         # print(messages)
#         samples.append(dict(conversations=messages))
#
#     # exit(0)
#     train_dataset = dataset_cls(samples, tokenizer=tokenizer, max_len=max_len)
#
#     if data_args.eval_data_path:
#         eval_json = json.load(open(data_args.eval_data_path, "r"))
#         eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
#     else:
#         eval_dataset = None
#
#     return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# @dataclass
# class DataArguments:
#     # data_path='./data/Belle_sampled_qwen.json'
#     data_path = '/data/data/med_dataset/HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl'
#     eval_data_path = None
#     lazy_preprocess = True


def load_dataset(max_len=256):
    # data_args = DataArguments()

    model_max_length = max_len
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/data/data/Qwen2-0.5B-Instruct",
        # cache_dir=training_args.cache_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    # tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(tokenizer.eos_token)
    print(tokenizer.special_tokens_map_extended)
    '''
    # EOS
    https://github.com/QwenLM/Qwen2.5/issues/517
    https://github.com/QwenLM/Qwen2.5/issues/33
    '''
    print(tokenizer.convert_tokens_to_ids("<|endoftext|>"))

    print(f'n_vocab: {tokenizer.vocab_size}')

    # tokenizer.im_start_token = '<|im_start|>'
    # tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    # tokenizer.im_end_token = '<|im_end|>'
    # tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # # Load data
    # data_module = make_supervised_data_module(
    #     tokenizer=tokenizer, data_args=data_args, max_len=model_max_length
    # )

    train_dataset = PretrainDataset(raw_data=[
        '/data/data/llm_data/raw_data_gpt2/medium-345M-k40.train.jsonl',
        # '/data/data/llm_data/raw_data_gpt2/webtext.train.jsonl',
        # '/data/data/llm_data/raw_data_gpt2/xl-1542M.train.jsonl',
        # '/data/data/llm_data/raw_data_gpt2/large-762M-k40.train.jsonl',
        # '/data/data/llm_data/raw_data_gpt2/large-762M.train.jsonl'

    ],
        tokenizer=tokenizer,
        max_len=max_len,
        # cache='/data/data/llm_data/pretrain_20240920.pkl'
        # cache='/data/data/llm_data/pretrain_2024094_slim_medium_345M_k40.pkl'
        cache='/data/data/llm_data/pretrain_2024094_slim_medium_345M_k40_s256.pkl'
    )

    print(f'train_dataset_len: {len(train_dataset)}')

    # print(f'n_train: {len(data_module["train_dataset"])}')
    # # print(f'n_val: {len(data_module["eval_dataset"])}')

    data_module = dict(train_dataset=train_dataset, eval_dataset=None)
    return data_module


if __name__ == '__main__':
    load_dataset()
