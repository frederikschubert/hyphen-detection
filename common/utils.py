import multiprocessing as mp
import os
from random import shuffle
from typing import List

import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


def imap_progress(f, iterable: List, flatten=False):
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(f, iterable), total=len(iterable)):
            if result:
                if flatten:
                    results.extend(result)
                else:
                    results.append(result)
    return results


def get_train_val_samplers(dataset: Dataset, val_split: float = 0.2, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler