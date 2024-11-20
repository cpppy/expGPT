import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import transformers
from typing import List, Dict
import copy

from dschat.utils.utils import print_rank_0
# from finetune_qwen.datasets.load_dataset_private_qa import preprocess


def example_by_gpt4():
    # 假设你的parquet文件路径为'data/dataset.parquet'
    dataset_path = '/data/data/llm_datasets/medmcqa/data/train-00000-of-00001.parquet'

    # 使用pandas读取parquet文件
    df = pd.read_parquet(dataset_path)

    # 现在df包含了parquet文件的内容
    # 下一步取决于你怎么想转换这个数据来适应你的LLM模型
    # 例如，你可能需要构建一个特定格式的'instruct'数据集，依据模型的需求。

    # 举例，如果你的模型需要"text"和"summary"这样的列，可以假设原始数据集有这些列，或者你需要从现有数据中生成这些内容。
    # 如果df已经有了需要的列，可以直接开始使用或者进一步处理数据。

    # 输出前几行看数据结构
    print(df.head())

    # 假设df已经有了"text"和"summary"列，
    # 你可以直接使用这个DataFrame进行模型训练，或进一步转换为模型所需要的格式

    # 例子：转换DataFrame为datasets库中的Dataset格式，方便模型微调使用
    from datasets import Dataset

    # 通过DataFrame创建Hugging Face dataset
    hf_dataset = Dataset.from_pandas(df)

    # 查看转换后的数据集
    print(hf_dataset)

    # 此时hf_dataset就可以用于LLM的微调任务

"""
https://hf-mirror.com/datasets/openlifescienceai/medmcqa

Data Fields
    id : a string question identifier for each example
    question : question text (a string)
    opa : Option A
    opb : Option B
    opc : Option C
    opd : Option D
    cop : Correct option, i.e., 1,2,3,4
    choice_type ({"single", "multi"}): Question choice type.
    "single": Single-choice question, where each choice contains a single option.
    "multi": Multi-choice question, where each choice contains a combination of multiple suboptions.
    exp : Expert's explanation of the answer
    subject_name : Medical Subject name of the particular question
    topic_name : Medical topic name from the particular subject

"""

# def preprocess(messages, tokenizer, max_len):
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         return_tensors="pt",
#     )
#     print(input_ids)
#
#     exit(0)


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
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_fixed(
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
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    # print(input_ids)
    # print(targets)
    # exit(0)

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_fixed_simple(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    # roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    #
    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id
    #
    # nl_tokens = tokenizer('\n').input_ids
    # _system = tokenizer('system').input_ids + nl_tokens
    # _user = tokenizer('user').input_ids + nl_tokens
    # _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        # if roles[source[0]["from"]] != roles["user"]:
        #     source = source[1:]

        prompt = tokenizer.apply_chat_template(
            conversation=source,
            # add_generation_prompt=True,
            return_tensors='pt',
            tokenize=False,
        )
        print(prompt)
        exit(0)

        input_id, target = [], []

        # # TODO: whether use system_prompt?
        # system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        # input_id += system

        # target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        # assert len(input_id) == len(target)

        print(source)
        exit(0)

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
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    # print(input_ids)
    # print(targets)
    # exit(0)

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



class MedMCQ(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int):
        super(MedMCQ, self).__init__()

        print_rank_0("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess_fixed_simple(sources, tokenizer, max_len)

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


def create_optimized_prompt(row):
    # 构造解释性更强、指示性更明确的格式
    # prompt = (f"Read the question and choose the correct option based on the options given.\n\n"
    #           f"Question: {row['question']}\n"
    #           f"Options:\n"
    #           f"A. {row['A']}\n"
    #           f"B. {row['B']}\n"
    #           f"C. {row['C']}\n"
    #           f"D. {row['D']}\n"
    #           f"\nThe correct option is: ")
    prompt = (f"Read the question and choose the correct option based on the options given.\n\n"
              f"Question: {row['question']}\n"
              f"Options:\n"
              f"A. {row['opa']}\n"
              f"B. {row['opb']}\n"
              f"C. {row['opc']}\n"
              f"D. {row['opd']}\n"
              f"\nThe correct option is: ")
    return prompt


def main():

    samples = []
    data_paths = [
        '/data/data/llm_datasets/medmcqa/data/train-00000-of-00001.parquet',
    ]
    for data_path in data_paths:
        print_rank_0(f'data_path: {data_path}')
        df = pd.read_parquet(data_path)
        df['optimized_prompt'] = df.apply(create_optimized_prompt, axis=1)
        df['answer'] = df.apply(lambda x: 'ABCDEFGH'[x['cop']], axis=1)
        df = df[['optimized_prompt', 'answer']]
        print(f'n_raw_sample: {df.shape[0]}')
        raw_datas = df.to_dict(orient='records')[0:50000]
        # print(raw_datas[0])
        # exit(0)
        choice_types = set([x['answer'] for x in raw_datas])
        print_rank_0(f'unique_choice: {choice_types}')

        for data in raw_datas:
            try:
                data = {k.strip(): v.strip() for k, v in data.items()}
                messages = []
                messages.append({"role": "user", "content": data['optimized_prompt']})
                messages.append({"role": "assistant", "content": data['answer']
                                 })
                # print(messages)
                samples.append(dict(conversations=messages))
            except Exception as e:
                print(f'[WARNING] Fail to parse sample: {data}')
                print(e)
                continue

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/data/Qwen2.5-0.5B-Instruct",
        # cache_dir=training_args.cache_dir,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

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

    samples = samples[0:10]

    max_len=512
    train_dataset = MedMCQ(samples, tokenizer=tokenizer, max_len=max_len)
    print_rank_0(f'n_train_sample: {len(train_dataset)}')

    # for s in train_dataset:
    #     # print(s['labels'][0])
    #     # print(tokenizer.decode(s['input_ids'], skip_special_tokens=False))
    #
    #     print(tokenizer.decode(s['input_ids'][s['labels']>0], skip_special_tokens=False))
    #
    #     # print(s['attention_mask'])
    #
    #     # exit(0)

    return train_dataset


if __name__=='__main__':

    main()

