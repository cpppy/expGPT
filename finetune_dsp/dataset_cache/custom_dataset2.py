import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import transformers
from typing import List, Dict
import copy

from dschat.utils.utils import print_rank_0
# from finetune_qwen.datasets.load_dataset_private_qa import preprocess

from tqdm import tqdm
import glob
import os
import random


"""
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant

"""

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

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
        # if roles[source[0]["from"]] != roles["user"]:
        #     source = source[1:]

        input_id, target = [], []

        # TODO: whether use system_prompt?
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system

        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
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

    # print(input_ids)
    # print(targets)
    # exit(0)

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int64)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_fixed(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        pad_and_clip: bool=True,
        output_tensor: bool=True,
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
    pbar = tqdm(total=len(sources), desc='raw_text_processing')
    for i, source in enumerate(sources):
        # if roles[source[0]["from"]] != roles["user"]:
        #     source = source[1:]

        input_id, target = [], []

        # # TODO: whether use system_prompt?
        # system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        # input_id += system

        # target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        # assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            # _input_id = tokenizer(role).input_ids + nl_tokens + \
            #             tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
            # input_id += _input_id
            # if role == '<|im_start|>user':
            #     _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            # elif role == '<|im_start|>assistant':
            #     _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
            #               _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            # else:
            #     raise NotImplementedError

            if sentence['role'] == 'user':
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                            tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif sentence['role'] == 'assistant':
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                            tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
                _target = copy.deepcopy(_input_id)
                _target[0: len(tokenizer(role).input_ids + nl_tokens)] = [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids + nl_tokens)
            else:
                raise NotImplementedError

            input_id += _input_id
            target += _target
        assert len(input_id) == len(target)

        ################## padding or clip ##################
        if pad_and_clip:
            input_id += [tokenizer.pad_token_id] * max(0, max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * max(0, max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        else:
            input_ids.append(input_id)
            targets.append(target)

        pbar.update(1)
    pbar.close()

    if output_tensor:
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.int)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
    else:
        attention_mask = [[]] * len(input_ids)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )


class CustomQADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int,
                 datasetname: str):
        super(CustomQADataset, self).__init__()
        self.datasetname = datasetname

        print_rank_0("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess_fixed(sources,
                                     tokenizer,
                                     max_len,
                                     pad_and_clip=False,
                                     output_tensor=False)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor] or Dict[str, List]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def build_dataset(tokenizer, max_len=256, eval_ratio=0.1):

    # samples = []
    # # data_paths = [
    # #     '/data/data/llm_data/Zhihu-KOL/data/train-00000-of-00005-a1278ede4e8c5cdb.parquet',
    # # ]
    # raw_dir = '/data/data/llm_data/Zhihu-KOL'
    # datasetname = os.path.basename(raw_dir).lower()
    #
    # data_paths = glob.glob(f'{raw_dir}/data/*.parquet')#[0:1]
    # data_paths = sorted(data_paths)
    # for data_path in data_paths:
    #     print_rank_0(f'data_path: {data_path}')
    #     df = pd.read_parquet(data_path)
    #     # df['optimized_prompt'] = df.apply(create_optimized_prompt, axis=1)
    #     # df['answer'] = df.apply(lambda x: 'ABCDEFGH'[x['cop']], axis=1)
    #     # df = df[['optimized_prompt', 'answer']]
    #     print(f'n_raw_sample: {df.shape[0]}')
    #     raw_datas = df.to_dict(orient='records')#[0:500]
    #     # print(raw_datas[0])
    #     # exit(0)
    #     # choice_types = set([x['answer'] for x in raw_datas])
    #     # print_rank_0(f'unique_choice: {choice_types}')
    #
    #     for data in raw_datas:
    #         try:
    #             # data = {k.strip(): v.strip() for k, v in data.items()}
    #             messages = []
    #             messages.append({"role": "user", "content": data['INSTRUCTION']})
    #             messages.append({"role": "assistant", "content": data['RESPONSE']
    #                              })
    #             # print(messages)
    #             samples.append(dict(conversations=messages))
    #         except Exception as e:
    #             print(f'[WARNING] Fail to parse sample: {data}')
    #             print(e)
    #             continue

    samples = read_raw_data()
    datasetname = 'medical_mix_part1'

    random.seed(1234)
    random.shuffle(samples)
    n_eval = min(int(len(samples) * (1 - eval_ratio)), 5000)
    split_pos = len(samples) - n_eval

    train_dataset = CustomQADataset(samples[:split_pos],
                                    tokenizer=tokenizer,
                                    max_len=max_len,
                                    datasetname=datasetname,
                                    )
    print_rank_0(f'n_train_sample: {len(train_dataset)}')

    # for s in train_dataset:
    #     print(len(s['input_ids']), len(s['labels']), len(s['attention_mask']))
    #     print(s['input_ids'])
    #     print(s['labels'])
    #     exit(0)

    eval_dataset = CustomQADataset(samples[split_pos:],
                                   tokenizer=tokenizer,
                                   max_len=max_len,
                                   datasetname=datasetname)
    print_rank_0(f'n_eval_sample: {len(eval_dataset)}')

    return train_dataset, eval_dataset


import json
def load_texts(data_file, expected_size=None):
    texts = []
    for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line))
    return texts

def read_raw_data():
    data_paths = [
        '/data/data/med_dataset/medical/finetune/train_zh_0.json',
        '/data/data/med_dataset/medical/finetune/train_en_1.json',
    ]
    eval_data_paths = [
        '/data/data/med_dataset/medical/finetune/valid_zh_0.json',
        '/data/data/med_dataset/medical/finetune/valid_en_1.json',
    ]
    data_paths = data_paths + eval_data_paths

    samples = []
    for data_path in data_paths:
        print(f'data_path: {data_path}')
        raw_datas = load_texts(data_path)  # [0:10]
        for data in raw_datas:
            messages = []
            messages.append({"from": "user", "value": data['instruction'] + data['input']})
            messages.append({"from": "assistant", "value": data['output']
                             })
            # print(messages)
            samples.append(dict(conversations=messages))
    print(f'n_samples: {len(samples)}')
    return data_paths



if __name__=='__main__':

    read_raw_data()
    exit(0)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # pretrained_model_name_or_path="/data/Qwen2.5-0.5B-Instruct",
        pretrained_model_name_or_path="/data/Qwen/Qwen2-0.5B-Instruct",
        # cache_dir=training_args.cache_dir,
        # model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # print(tokenizer.pad_token, tokenizer.pad_token_id)
    # exit(0)

    # prompt = "Give me a short introduction to large language model."
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # print(text)
    # # model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # samples = samples[0:10]

    build_dataset(tokenizer, max_len=256)

