import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
import pandas as pd
import glob

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

        # from finetune_dsp.datasets import load_medmcq_adv
        # self.raw_datasets = {'train': load_medmcq_adv.main()}

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# Chinese dataset
class Wangrui6ZhihuKOLDatasetLocal(object):

    def __init__(self, output_path, seed, local_rank, raw_data_dir):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.raw_data_dir = raw_data_dir

        self.raw_datasets = {'train': self._load_raw_samples()}

        self.dataset_name = "wangrui6/Zhihu-KOL"
        # self.dataset_name = dataset_name
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"

    def get_train_data(self):
        from dschat.utils.data.data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank,
                                            self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from dschat.utils.data.data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return " Human: " + sample['INSTRUCTION'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return " " + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return " Human: " + sample[
                'INSTRUCTION'] + " Assistant: " + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def _load_raw_samples(self):
        samples = []

        data_paths = glob.glob(f'{self.raw_data_dir}/data/*.parquet')#[0:1]
        data_paths = sorted(data_paths)
        for data_path in data_paths:
            # print_rank_0(f'data_path: {data_path}')
            logger.debug(f'data_path: {data_path}')
            df = pd.read_parquet(data_path)
            # df['optimized_prompt'] = df.apply(create_optimized_prompt, axis=1)
            # df['answer'] = df.apply(lambda x: 'ABCDEFGH'[x['cop']], axis=1)
            # df = df[['optimized_prompt', 'answer']]
            # logger.debug(f'n_raw_sample: {df.shape[0]}')
            df = df[['INSTRUCTION', 'RESPONSE']]
            _samples = df.to_dict(orient='records')#[0:100]
            # print(raw_datas[0])
            # exit(0)
            # choice_types = set([x['answer'] for x in raw_datas])
            # print_rank_0(f'unique_choice: {choice_types}')

            # for data in raw_datas:
            #     try:
            #         data = {k.strip(): v.strip() for k, v in data.items()}
            #         messages = []
            #         messages.append({"role": "user", "content": data['optimized_prompt']})
            #         messages.append({"role": "assistant", "content": data['answer']
            #                          })
            #         # print(messages)
            #         samples.append(dict(conversations=messages))
            #     except Exception as e:
            #         print(f'[WARNING] Fail to parse sample: {data}')
            #         print(e)
            #         continue
            samples.extend(_samples)
        return samples



def main():

    dataset = Wangrui6ZhihuKOLDatasetLocal(
        output_path='/mnt2/data/dsp_data_files',
        seed=1234,
        local_rank=-1,
        raw_data_dir="/data/data/llm_data/Zhihu-KOL",
    )
    print(f'n_train_dataset: {len(dataset.get_train_data())}')
    print(f'n_eval_dataset: {len(dataset.get_eval_data())}')




if __name__=='__main__':

    main()


