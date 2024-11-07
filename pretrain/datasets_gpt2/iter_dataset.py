import itertools

import torch
from torch.utils.data import IterableDataset, DataLoader

import numpy as np
from copy import deepcopy


class CustomIterableDatasetv1(IterableDataset):

    def __init__(self, filename, buffersize, seed, shuffle=False):

        # Store the filename in object's memory
        self.filename = filename
        self.buffer_size = buffersize
        self.generator = np.random.default_rng(seed=seed)
        self.shuffle = shuffle

    def preprocess(self, text):

        ### Do something with text here
        text_pp = text.lower().strip('\n')
        ###

        return text_pp

    def line_mapper(self, line):

        # Splits the line into text and label and applies preprocessing to the text
        text, label = line.split('-')
        text = self.preprocess(text)
        label = self.preprocess(label)

        return text, label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1

        # Create an iterator
        file_itr = open(self.filename)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        # Add multiworker functionality
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        if self.shuffle:
            return self._shuffle(mapped_itr)
        else:
            return mapped_itr

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def _shuffle(self, ex_iterable):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer



if __name__=='__main__':
    # base_dataset = CustomIterableDatasetv1("iter_ds.txt",4,1,False)
    base_dataset = CustomIterableDatasetv1("iter_ds.txt", 4, 1, True)
    dataloader = DataLoader(base_dataset, batch_size=3, num_workers=0)
    for X, y in dataloader:
        print(X, y)
