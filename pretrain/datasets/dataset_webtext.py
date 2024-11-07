import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

import os
import glob

from pretrain.datasets.packed_dataset import PackedDataset, CombinedDataset

from loguru import logger


def create_dataloader(
        batch_size: int,
        block_size: int,
        data_dir: str,
        fabric,
        shuffle: bool = True,
        seed: int = 12345,
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
        dataset = PackedDataset(
            filenames, n_chunks=4, block_size=block_size, shuffle=shuffle, seed=seed,
            num_processes=fabric.world_size, process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    # weights = [weight for _, weight in data_config]
    # sum_weights = sum(weights)
    # weights = [el / sum_weights for el in weights]
    # combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    # return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def build_dataset(data_dir='/data/data/llm_data/pretrain_webtext',
                  prefix='webtext.train',
                  block_size=2049,
                  shuffle=False,
                  world_size=1,
                  global_rank=0,
                  seed=1234,
                  ):
    filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
    logger.info(f'FOUND {len(filenames)} files in {prefix}')
    dataset = PackedDataset(
        filenames, n_chunks=4, block_size=block_size, shuffle=shuffle, seed=seed,
        num_processes=world_size, process_rank=global_rank,
    )
    return dataset


if __name__ == '__main__':

    dataset = build_dataset()
    print(f'n_data_block: {len(dataset._filenames)}')

    combined_dataset = CombinedDataset([dataset], seed=1234)
    print(f'dataset_len: {len(dataset)}')

    batch_size = 4
    dataloader = DataLoader(combined_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=False)
    print(f'n_batch: {len(dataloader)}')

    for batch in dataloader:
        print(batch.shape)
        print(list(batch[0]))
        print(list(batch[1]))

        break
