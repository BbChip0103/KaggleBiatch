import os
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
log = logging.getLogger(__name__)


class Splitter(object):
    def __init__(self, config, train):
        self.config = config
        self.train = train
        self.splits_path = os.path.join(self.config['splits_path'], "splits.pth")
        self.splits = {"train": [], "val": []}

    def __call__(self):
        log.info("Splitting")
        if (Path(self.splits_path).is_file()):
            log.info(f"Loading splits from {self.splits_path}")
            self.splits = torch.load(self.splits_path)
        else:
            log.info(f"Getting splits by hand.")
            keys = list(self.train.keys())
            targets = [self.train[k]['target'] for k in keys]
            if self.config['competition_type'] != "multilabel":
                split_gen = self.split_gen_ordinary(keys=keys, targets=targets)
            else:
                split_gen = self.split_gen_multilabel(keys=keys, targets=targets)

            for i, (train_indices, val_indices) in enumerate(split_gen):
                self.splits['train'].append([keys[i] for i in train_indices])
                self.splits['val'].append([keys[i] for i in val_indices])
            torch.save(self.splits, self.splits_path)

        self.assert_validity()
        return self.splits

    def split_gen_ordinary(self, keys, targets):
        log.info("Splitting ordinary")
        if self.config['dataset']['stratified']:
            kf = StratifiedKFold
        else:
            kf = KFold
        kf = kf(n_splits=self.config['dataset']["n_folds"],
                random_state=self.config['dataset']["RS"],
                shuffle=self.config["dataset"]['shuffle'])
        return kf.split(keys, targets)

    def split_gen_multilabel(self, keys, targets):
        log.info("Splitting multilabel")
        all_inds = list(range(len(targets)))
        val_folds = self.mass_prob_split(np.array(targets), folds=self.config['dataset']['n_folds'])
        train_folds = [list(set(all_inds)-set(list(fold))) for fold in val_folds]
        return zip(train_folds, val_folds)

    def assert_validity(self):
        for fold in range(self.config['dataset']['n_folds']):
            train_inds = self.splits['train'][fold]
            val_inds = self.splits['val'][fold]
            assert self.lists_overlap(train_inds, val_inds) == 0
            assert len(train_inds) + len(val_inds) == len(self.train), f"{len(train_inds)}, {len(val_inds)}, {len(self.train)}"
            log.info(f"Fold {fold}, {len(train_inds)} / {len(val_inds)}")
        log.info("Splits validity is OK.")

    @staticmethod
    def lists_overlap(a, b):
        sb = set(b)
        return any(el in sb for el in a)

    @staticmethod
    # https://stats.stackexchange.com/questions/65828/how-to-use-scikit-learns-cross-validation-functions-on-multi-label-classifiers
    def mass_prob_split(y, folds):
        obs, classes = y.shape
        dist = y.sum(axis=0).astype('float')
        dist /= dist.sum()
        index_list = []
        fold_dist = np.zeros((folds, classes), dtype='float')
        for _ in np.arange(folds):
            index_list.append([])
        for i in np.arange(obs):
            if i < folds:
                target_fold = i
            else:
                normed_folds = fold_dist.T / fold_dist.sum(axis=1)
                how_off = normed_folds.T - dist
                target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
            fold_dist[target_fold] += y[i]
            index_list[target_fold].append(i)
        return index_list
