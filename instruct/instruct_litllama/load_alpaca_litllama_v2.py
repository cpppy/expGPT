import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data

def get_batch(data: list):
    micro_batch_size = 4
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    # x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_data_for_train():
    train_set, val_set = load_datasets(data_dir='/data/lit-llama/recon/data/alpaca')
    print(f'n_train: {len(train_set)}')
    print(f'n_val: {len(val_set)}')
    x, y = get_batch(train_set)

    print(x.shape, y.shape)
    print(x[0])
    print(y[0])


class AlpacaDataset(Dataset):

    def __init__(self,
                 data_dir='/data/lit-llama/recon/data/alpaca',
                 mode='train'):
        super(AlpacaDataset, self).__init__()
        if mode == 'train':
            self.samples = torch.load(os.path.join(data_dir, "train.pt"))
        else:
            self.samples = torch.load(os.path.join(data_dir, "test.pt"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"].type(torch.int64)
        labels = self.samples[idx]["labels"].type(torch.int64)
        # print(f'input_ids: {input_ids.shape}')
        # print(f'labels: {labels.shape}')
        return input_ids, labels

    @staticmethod
    def collate_fn(batch):
        input_ids, labels = zip(*batch)  # transposed
        max_len = max(len(s) for s in input_ids)
        def pad_right(x, pad_id):
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
        x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
        y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
        return x, y


if __name__=='__main__':

    # load_data_for_train()

    dataset = AlpacaDataset()
    s = dataset.__getitem__(1)

    dataloader = DataLoader(dataset,
                            batch_size=2,
                            num_workers=1,
                            drop_last=True,
                            collate_fn=dataset.collate_fn)

    for batch in dataloader:
        inputs, labels = batch
        print(inputs.shape)
        print(labels.shape)

        break







