import logging
from typing import Callable
from typing import Collection
from typing import Iterator

import numpy as np
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class MultipleDatasetIterFactory(AbsIterFactory):
    def __init__(
        self,
        build_funcs,
        seed: int = 0,
        shuffle: bool = False,
        dataset_training_portion: dict = {}
    ):
        assert check_argument_types()
        self.build_funcs = build_funcs
        self.seed = seed
        self.shuffle = shuffle
        self.dataset_training_portion = dataset_training_portion

    def build_iter(self, epoch: int, shuffle: bool = None, collate_fn=None) -> Iterator:

        dataset_index = np.random.choice(len(self.dataset_training_portion),p=list(self.dataset_training_portion.values()))
        dataset = list(self.dataset_training_portion.keys())[dataset_index]
        build_func = self.build_funcs[dataset]

        logging.info(f"Building iter-factory for Dataset:{dataset}")
        iter_factory = build_func()
        assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
        return iter_factory.build_iter(epoch, shuffle, collate_fn=collate_fn)
