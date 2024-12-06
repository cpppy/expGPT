import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string
import time
import numpy as np

import lmdb
import pickle

import torch
from torch.utils.data.dataset import Dataset


class LMDBSearch(object):
    def __init__(self, db_path, max_readers=8):
        super(LMDBSearch, self).__init__()
        self.db_path = db_path
        self.max_readers = max_readers

        # prepare initialization
        env = lmdb.open(db_path, subdir=osp.isdir(db_path), max_readers=max_readers,
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
        # print(env.info())
        with env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            # self.length = pa.deserialize(txn.get(b'__len__'))
            # self.keys = pa.deserialize(txn.get(b'__keys__'))
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        print(f'lmdb_len: {self.length}')
        print(f'n_key: {len(self.keys)}')
        print('[DATASET] load data from lmdb file, n_sample: {}'.format(self.length))
        env.close()

    def _open_lmdb(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), max_readers=self.max_readers,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def search_by_index(self, key):
        # key = index.encode()
        assert key in self.keys

        if not hasattr(self, 'txn'):
            self._open_lmdb()

        byteflow = self.txn.get(key)
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked
        return imgbuf

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'



IGNORE_TOKEN_ID = -100
PAD_TOKEN_ID = 151643

class CustomDPODataset(Dataset):

    def __init__(self, db_path, max_len=256):
        super(CustomDPODataset, self).__init__()
        assert db_path is not None
        self.lmdb_search = LMDBSearch(db_path=db_path)
        self.keys = self.lmdb_search.keys
        self.len = self.lmdb_search.length
        self.max_len = max_len

    def __len__(self):
        return self.len

    def _pad_and_clip(self, sample):
        input_ids = sample['input_ids']
        labels = sample['labels']
        max_len = self.max_len
        ################## padding or clip ##################
        input_ids += [PAD_TOKEN_ID] * max(0, max_len - len(input_ids))
        labels += [IGNORE_TOKEN_ID] * max(0, max_len - len(labels))
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.int)
        labels = torch.tensor(labels, dtype=torch.int)
        attention_mask = input_ids.ne(PAD_TOKEN_ID)

        return dict(
            input_ids=input_ids.unsqueeze(dim=0),
            labels=labels.type(torch.int64).unsqueeze(dim=0),
            attention_mask=attention_mask.unsqueeze(dim=0),
        )


    '''
        elif self.train_phase == 2:
        return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
               self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]

    '''

    def __getitem__(self, idx):
        sample_pairs = self.lmdb_search.search_by_index(self.keys[idx])
        chosen = self._pad_and_clip(sample_pairs['chosen'])
        rejected = self._pad_and_clip(sample_pairs['rejected'])
        return (
            chosen["input_ids"],
            chosen["attention_mask"],
            rejected["input_ids"],
            rejected["attention_mask"]
        )




if __name__ == '__main__':

    search_api = LMDBSearch(
        db_path='/mnt2/data/dsp_data_files2/dpo_zh_500_train_train_tokenizerQwen25_cache_20241204.lmdb'
    )

    print(search_api.keys[0:10])

    # img_fn = search_api.keys[101].decode()
    # print(img_fn)
    # result = search_api.search_by_index('1')
    # print(result)
    # # cv2.imwrite(img_fn, img_cv2)
    #
    # print(search_api.length)
    # print(search_api.keys[0:10])

    dataset = CustomDPODataset(
        db_path='/mnt2/data/dsp_data_files2/dpo_zh_500_train_train_tokenizerQwen25_cache_20241204.lmdb'
    )
    print(dataset.__getitem__(0))

