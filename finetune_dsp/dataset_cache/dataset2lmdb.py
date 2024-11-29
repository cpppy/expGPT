import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string
import time

import lmdb
import pickle
import tqdm
# import pyarrow as pa

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

from glob import glob

from loguru import logger

img_formats = ['parquet']  # acceptable image suffixes


class RawDataset(Dataset):

    def __init__(self, data_dir):
        img_files = glob(f'{data_dir}/*/*/*.*')
        self.img_files = sorted(
            [
                x
                for x in img_files
                if x.split('.')[-1].lower() in img_formats
            ])
        logger.info(f'FOUND {len(self.img_files)} images in {data_dir}')
        logger.info(f'{self.img_files[0]}')

        geo_dir = f'{data_dir}/geo_data'
        geo_files = glob(f'{geo_dir}/*.txt')
        self.img_files.extend(geo_files)


    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def raw_reader(path):
        with open(path, 'rb') as f:
            bin_data = f.read()
        return bin_data

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_fn = os.path.basename(img_file)
        img_data = self.raw_reader(img_file)
        return dict(filename=img_fn, raw=img_data)


class Dataset2Lmdb(object):

    def __init__(self, dataset, lmdb_path, key_tag, num_workers=4):
        super(Dataset2Lmdb, self).__init__()
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=1,
                                      num_workers=num_workers,
                                      collate_fn=lambda x: x)
        self.lmdb_path = lmdb_path
        self.key_tag = key_tag

    @staticmethod
    def dumps_pyarrow(obj):
        """
        Serialize an object.

        Returns:
            Implementation-dependent bytes-like object
        """
        # return pa.serialize(obj).to_buffer()
        return pickle.dumps(obj)

    # def generate(self, write_frequency=5000):
    #     print("Generate LMDB to %s" % self.lmdb_path)
    #     isdir = os.path.isdir(self.lmdb_path)
    #     db = lmdb.open(self.lmdb_path, subdir=isdir,
    #                    map_size=1099511627776, readonly=False,
    #                    meminit=False, map_async=True)
    #
    #     pbar = tqdm.tqdm(desc='write2lmdb', total=len(self.data_loader))
    #     txn = db.begin(write=True)
    #     for idx, batch in enumerate(self.data_loader):
    #         sample = batch[0]
    #         # print(type(data), data)
    #         img_fn = sample['filename']
    #         img_data = sample['raw']
    #         txn.put(u'{}'.format(img_fn).encode('ascii'), self.dumps_pyarrow(img_data))
    #         if idx % write_frequency == 0:
    #             print("[%d/%d]" % (idx, len(self.data_loader)))
    #             txn.commit()
    #             txn = db.begin(write=True)
    #         pbar.update(1)
    #     pbar.close()
    #
    #     # finish iterating through dataset
    #     txn.commit()
    #
    #     keys = [u'{}'.format(os.path.basename(x)).encode('ascii') for x in self.dataset.img_files]
    #     with db.begin(write=True) as txn:
    #         txn.put(b'__keys__', self.dumps_pyarrow(keys))
    #         txn.put(b'__len__', self.dumps_pyarrow(len(keys)))
    #
    #     print("Flushing database ...")
    #     db.sync()
    #     db.close()

    def generate_for_text(self, write_frequency=5000):
        print("Generate LMDB to %s" % self.lmdb_path)
        isdir = os.path.isdir(self.lmdb_path)
        db = lmdb.open(self.lmdb_path, subdir=isdir,
                       map_size=1099511627776, readonly=False,
                       meminit=False, map_async=True)

        pbar = tqdm.tqdm(desc='write2lmdb', total=len(self.data_loader))
        txn = db.begin(write=True)
        for idx, batch in enumerate(self.data_loader):
            sample = batch[0]
            # print(type(data), data)
            input_ids = sample['input_ids']
            labels = sample['labels']
            sample_data = dict(input_ids=input_ids, labels=labels)
            txn.put(u'{}'.format(f'{self.key_tag}_idx_{idx}').encode('ascii'), self.dumps_pyarrow(sample_data))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(self.data_loader)))
                txn.commit()
                txn = db.begin(write=True)
            pbar.update(1)
        pbar.close()

        # finish iterating through dataset
        txn.commit()

        keys = [u'{}'.format(f'{self.key_tag}_idx_{i}').encode('ascii') for i in range(len(self.dataset))]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', self.dumps_pyarrow(keys))
            txn.put(b'__len__', self.dumps_pyarrow(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()


def trans_dataset_to_lmdb():

    import transformers
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

    # from finetune_dsp.dataset_cache.custom_dataset import build_dataset
    # train_dataset, eval_dataset = build_dataset(tokenizer=tokenizer, eval_ratio=0.05)
    from finetune_dsp.dataset_cache.custom_dataset_multi import build_dataset
    train_dataset, eval_dataset = build_dataset(eval_ratio=0.05)

    ds2lmdb_tool = Dataset2Lmdb(dataset=train_dataset,
                                lmdb_path=f'/mnt2/data/dsp_data_files2/{train_dataset.datasetname}_train_tokenizerQwen25_cache_20241126.lmdb',
                                key_tag=train_dataset.datasetname,
                                num_workers=4)
    ds2lmdb_tool.generate_for_text()

    ds2lmdb_tool = Dataset2Lmdb(dataset=eval_dataset,
                                lmdb_path=f'/mnt2/data/dsp_data_files2/{eval_dataset.datasetname}_eval_tokenizerQwen25_cache_20241126.lmdb',
                                key_tag=eval_dataset.datasetname,
                                num_workers=4)

    ds2lmdb_tool.generate_for_text()


if __name__ == "__main__":

    trans_dataset_to_lmdb()

