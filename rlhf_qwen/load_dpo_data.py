import os
import jsonlines
import json
from tqdm import tqdm

from typing import Dict

from torch.utils.data.dataset import Dataset

import sys
sys.path.append('..')

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import random


def load_texts(data_file, expected_size=None):
    texts = []
    for line in tqdm(jsonlines.open(data_file), total=expected_size, desc=f'Loading {data_file}'):
        texts.append(json.loads(line))
    return texts


def load_raw_data():
    data_path = '../data/dpo_zh_500.jsonl'

    samples = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            samples.append(json.loads(line))
    print(f'n_samples: {len(samples)}')

    # print(samples[0])
    print(json.dumps(samples[0], indent=4, ensure_ascii=False))
    return samples



class RawDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def shuffle(self):
        _samples = random.shuffle(self.samples)
        return RawDataset(_samples)


def build_dpo_dataset():

    # load raw data
    # raw_samples = load_raw_data()
    #
    # raw_dataset = RawDataset(raw_samples)

    from datasets import load_dataset
    raw_dataset = load_dataset(path='../data',
                               data_files='../data/dpo_zh_500.jsonl',
                               # data_files={
                               #     'train': '../data/dpo_zh_500.jsonl'
                               # },
                               split='train')

    print(len(raw_dataset))

    from rlhf_qwen.template import get_conv_template
    prompt_template = get_conv_template('qwen')

    def return_prompt_and_responses(examples) -> Dict[str, str]:
        """Load the paired dataset and convert it to the necessary format.

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

        Prompts are structured as follows:
          system_prompt + history[[q,a], [q,a]...] + question
        """
        prompts = []
        for system, history, question in zip(examples["system"], examples["history"], examples["question"]):
            system_prompt = system or ""
            history_with_question = history + [[question, '']] if history else [[question, '']]
            prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
        return {
            "prompt": prompts,
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

    # Preprocess the dataset
    train_dataset = None
    max_train_samples = 0

    raw_datasets = {'train': raw_dataset}

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets['train']
    max_train_samples = len(train_dataset)
    # if args.max_train_samples is not None and args.max_train_samples > 0:
    #     max_train_samples = min(len(train_dataset), args.max_train_samples)
    #     train_dataset = train_dataset.select(range(max_train_samples))

    logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
    tokenized_dataset = train_dataset.shuffle().map(
        return_prompt_and_responses,
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )


    '''
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=512, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
    
    '''
    # full_max_length = max_source_length + max_target_length
    full_max_length = 1024
    train_dataset = tokenized_dataset.filter(
        lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                  and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
    )
    logger.debug(f"Num train_samples: {len(train_dataset)}")
    logger.debug("First train example:")
    first_example = train_dataset[0]
    logger.debug(f"prompt:\n{repr(first_example['prompt'])}")
    logger.debug(f"chosen:\n{repr(first_example['chosen'])}")
    logger.debug(f"rejected:\n{repr(first_example['rejected'])}")


    eval_dataset = None
    max_eval_samples = 0

    return train_dataset, eval_dataset


if __name__ == '__main__':

    build_dpo_dataset()

