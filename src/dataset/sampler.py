import logging
import random

import numpy as np
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
log = logging.getLogger(__name__)


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.keys = list(data_source.keys())

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)


class RandomSampler(SequentialSampler):
    def __init__(self, data_source):
        super().__init__(data_source)

    def __iter__(self):
        return iter(np.random.permutation(self.keys).tolist())


class BalancedSampler(WeightedRandomSampler):
    def __init__(self, data_source):
        targets = np.array([x['target'] for x in data_source.values()])
        self.keys = list(data_source.keys())
        sum_ =  targets.sum(axis=0)
        weights = targets.shape[0] / sum_
        print_dict = {k: v for k, v in zip(sum_, weights)}
        log.info(f"Label x Weight coeff Sampler: {print_dict}")
        super().__init__(weights=weights, num_samples=len(targets))

    def __iter__(self):
        return iter(np.random.permutation(self.keys).tolist())


class ClassVsOthersSampler(Sampler):
    def __init__(self, data_source, class_ratio=0.05):
        self.data_source = data_source
        self.class_ratio = class_ratio
        self.positive_keys = [k for k, v in self.data_source.items() if v['target'] == 1]
        self.negative_keys = [k for k, v in self.data_source.items() if v['target'] == 0]
        self.get_length()

    def get_length(self):
        targets = [x['target'] for x in self.data_source.values()]
        self.postive_length = np.array(targets).sum()
        self.negative_length = int(self.postive_length * ((1 - self.class_ratio) / self.class_ratio))
        self.negative_length = min(self.negative_length, len(self.data_source) - self.postive_length)
        self.length = min(self.negative_length + self.postive_length, len(self.data_source))
        pos_ratio = self.postive_length / (self.postive_length + self.negative_length)
        log.info(f"Sampler ratio pos: {self.postive_length}, neg: {self.negative_length}, pos_ratio: {pos_ratio}")

    def __iter__(self):
        negative_keys = np.random.permutation(self.negative_keys).tolist()[:self.negative_length]
        res = self.positive_keys + negative_keys
        return iter(np.random.permutation(res).tolist())

    def __len__(self):
        return self.length
