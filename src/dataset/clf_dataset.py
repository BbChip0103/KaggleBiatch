import os
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import utils.utils as utils
log = logging.getLogger(__name__)


class ClfDataset(object):
    def __init__(self, config):
        self.config = config

    def read_data(self):
        self.data = {}

        for train_test_flag in ['train', 'test']:
            final_data = []
            self.ohe_cols = []
            log.info(f'Reading {train_test_flag}')
            for folder_name in tqdm(self.config['mode_stack']["folders"]):
                log.info(f"Processing folder {folder_name}")
                pred_load = torch.load(os.path.join(self.config['out_path'], folder_name, f"pred_{train_test_flag}.pth"))
                pred = pred_load['pred'][np.argsort(pred_load['id']),:]
                if self.config['debug']:
                    pred = pred[:self.config['debug_rows'], :]
                final_data.append(pred)

            self.data[train_test_flag] = np.concatenate(final_data, axis=1)
            log.info(f"Data shape {self.data[train_test_flag].shape}")

            sorted_ids = sorted(pred_load['id'])
            if self.config['debug']:
                sorted_ids = sorted_ids[:self.config['debug_rows']]
            self.data[f'{train_test_flag}_ids'] = sorted_ids

        # competiton specific
        # csv = pd.read_csv(os.path.join(self.config['data_folder'], "train.csv"))
        # csv.sort_values(self.config['competition_id_col'], inplace=True)
        # self.data['target'] = csv[self.config['competition_target_col']].values

        splits_data_path = os.path.join(str(Path.home()), ".kaggle_splits", "doodle")
        targets_path = os.path.join(splits_data_path, "val_targets.pth")
        if not Path(targets_path).is_file():
            with open(os.path.join(splits_data_path, "{}.pkl".format("val")), 'rb') as fp:
                val = pickle.load(fp)
            targets = []
            for k in sorted(val.keys()):
                targets.append(val[k]['word'])
            torch.save(np.array(targets).astype(np.float32), targets_path)
        else:
            targets = torch.load(targets_path)
        if self.config['debug']:
            targets = targets[:self.config['debug_rows']]
        log.info(f"Unique targets {np.unique(targets)}")
        self.data['target'] = targets

    def sort_by_id(self, data):
        return data['pred'][np.argsort(data['id']),:]

    def make_splits(self):
        pass
        # self.splits = {}
        # for i in np.arange(self.config['mode_stack']['n_folds']):
        #     self.splits[i] = {}
        #     for c in ['train', 'val']:
        #         self.splits[i][c] = utils.read_list_from_file(config.OUTPUT_FOLDER + folder + '/splits/%s_%d.txt'%(c, i))

    def __call__(self):
        self.read_data()
        self.make_splits()
