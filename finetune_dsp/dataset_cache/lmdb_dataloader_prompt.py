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


class CustomPromptDataset(Dataset):

    def __init__(self, db_path, max_len=256):
        super(CustomPromptDataset, self).__init__()
        assert db_path is not None
        self.lmdb_search = LMDBSearch(db_path=db_path)
        self.keys = self.lmdb_search.keys
        self.len = self.lmdb_search.length
        self.max_len = max_len

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        sample = self.lmdb_search.search_by_index(self.keys[idx])
        # print(sample)
        # exit(0)
        input_ids = sample['input_ids']
        labels = sample['labels']
        max_len = self.max_len

        """
        ################## padding or clip ##################
        input_ids += [151643] * max(0, max_len - len(input_ids))
        labels += [-100] * max(0, max_len - len(labels))
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.int)
        labels = torch.tensor(labels, dtype=torch.int)
        attention_mask = input_ids.ne(151643)
        """

        # only select prompt token_ids
        # print(input_ids)
        # print(labels)
        # exit(0)

        prompt_ids = [x for x, y in zip(input_ids, labels) if y == -100]
        ################## padding or clip ##################
        # pad to left
        prompt_ids = [151643] * max(0, max_len - len(prompt_ids)) + prompt_ids
        prompt_ids = prompt_ids[:max_len]

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
        attention_mask = prompt_ids.ne(151643)

        # return dict(
        #     input_ids=input_ids,
        #     # labels=labels.type(torch.int64),
        #     attention_mask=attention_mask,
        # )

        '''
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
        '''

        return prompt_ids, attention_mask, 151643




if __name__ == '__main__':

    # search_api = LMDBSearch(
    #     db_path='/mnt2/data/dsp_data_files2/zhihu-kol_train_tokenizerQwen2_cache_20241126.lmdb'
    # )
    #
    # print(search_api.keys[0:10])

    # img_fn = search_api.keys[101].decode()
    # print(img_fn)
    # result = search_api.search_by_index('1')
    # print(result)
    # # cv2.imwrite(img_fn, img_cv2)
    #
    # print(search_api.length)
    # print(search_api.keys[0:10])

    dataset = CustomPromptDataset(
        db_path='/mnt2/data/dsp_data_files2/zhihu-kol_train_tokenizerQwen2_cache_20241126.lmdb'
    )
    print(dataset.__getitem__(0))

