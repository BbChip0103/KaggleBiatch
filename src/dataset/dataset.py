import glob
import os
from pathlib import Path
import time
import json
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from dataset.splitter import Splitter
log = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, config):
        self.config = config

    def __call__(self):
        log.info('Reading dataframes..')
        self.read_train()
        self.read_test()
        self.splits = Splitter(config=self.config, train=self.train)()
        # if self.config['competition_type'] in ['multiclass', "multilabel"]:
        #     self.calculate_class_weights()
        self.class_weights = {}
        for i in range(5):
            self.class_weights[i] = torch.from_numpy(np.random.rand(28).astype(np.float32))

    def read_train(self, csv_name='train'):
        self.train = self.create_data_dict(type="train")

    def read_test(self, csv_name='test'):
        self.test = self.create_data_dict(type="test")

    def create_data_dict(self, type):
        log.info(f"Reading {type}...")
        csv_name = "train" if type == 'train' else "sample_submission"
        csv_path = os.path.join(self.config["data_folder"], type + ".csv")
        img_path = os.path.join(self.config["data_folder"], type)

        nrows = self.config['debug_rows'] if self.config["debug"] else None
        if (type == 'test') & (not Path(csv_path).is_file()):
            csv_path = os.path.join(self.config["data_folder"], "sample_submission" + ".csv")
        csv = pd.read_csv(csv_path, nrows=nrows)

        csv.rename(columns={self.config['competition_id_col']: "id"}, inplace=True)
        targets = None
        if "train" in type:
            csv.rename(columns={self.config['competition_target_col']: "target"}, inplace=True)
            csv, targets = self.map_categories(csv=csv)
            log.info(f"Mapping {self.categories_mapping}")

        csv['img_path'] = [[os.path.join(img_path, self.get_img_name(x, k, type))
                                         for k in ['red', 'green', 'blue', 'yellow']] \
                           for x in csv['id'].tolist()]
        return self.convert_pd_to_dict(df=csv, targets=targets)

    def get_img_name(self, x, k, type):
        img_extension = self.config['competition_img']['type'][type]
        return f"{x}_{k}.{img_extension}"

    def map_categories(self, csv):
        log.info("Mapping")
        mapping_path = os.path.join(self.config['splits_path'], "category_mapping.json")
        if Path("mapping_path").is_file():
            self.categories_mapping = json.load(open(mapping_path, "r"))
            return

        # if self.config['competition_type'] in (['multiclass', "binary"]):
        #     self.categories_mapping = {k: v for v, k in enumerate(sorted(csv['target'].unique().tolist()))}
        #     targets = [self.categories_mapping[k] for k in  csv['target'].tolist()]
        # elif self.config['competition_type'] in (["multilabel"]):
        #     targets = csv['target'].tolist()
        #     targets = [x.split(" ") for x in targets]
        #     res = list(set([y for elem in targets for y in elem]))
        #     if res[0].isdigit():
        #         self.categories_mapping = {k: v for v, k in enumerate(sorted(res, key=int))}
        #     else:
        #         self.categories_mapping = {k: v for v, k in enumerate(sorted(res))}
        #     for i in range(len(targets)):
        #         targets[i] = [self.categories_mapping[k] for k in targets[i]]
        #     one_hot = MultiLabelBinarizer()
        #     targets = one_hot.fit_transform(targets)
        # else:
        #     raise NotImplementedError

        targets = csv['target'].tolist()
        targets = [x.split(" ") for x in targets]
        res = list(set([y for elem in targets for y in elem]))
        if res[0].isdigit():
            self.categories_mapping = {k: v for v, k in enumerate(sorted(res, key=int))}
        else:
            self.categories_mapping = {k: v for v, k in enumerate(sorted(res))}
        for i in range(len(targets)):
            targets[i] = [self.categories_mapping[k] for k in targets[i]]
        one_hot = MultiLabelBinarizer()
        targets = one_hot.fit_transform(targets)

        csv.drop("target", 1, inplace=True)
        with open(mapping_path, "w") as f:
            json.dump(self.categories_mapping, f)
        return csv, targets

    def calculate_class_weights(self):
        self.class_weights = {}
        for fold_num in range(len(self.splits['train'])):
            data_source = {k: self.train[k] for k in self.splits['train'][fold_num]}
            targets = np.array([x['target'] for x in data_source.values()])
            sum_ =  targets.sum(axis=0)
            weights = targets.shape[0] / sum_
            weights = weights / sum(weights)
            self.class_weights[fold_num] = torch.from_numpy(weights.astype(np.float32))
            print_dict = {k: v for k, v in zip(sum_, weights)}
            log.info(f"Fold {fold_num}: N_labels x Weight coeff: {print_dict}.")

    def convert_pd_to_dict(self, df, targets):
        val_dict = {}
        final_dict = {}
        df = df.set_index("id")
        cols = df.columns.tolist()
        val_dict = {c: df[c].tolist() for c in cols}
        if targets is not None:
            if self.config['mode_target'] is not None:
                targets = targets[:, int(self.config['mode_target'])]
            val_dict.update({"target": targets})
            cols = cols + ["target"]
        for i, x in tqdm(enumerate(df.index.tolist())):
            dict_ = {x: {c: val_dict[c][i] for c in cols}}
            final_dict.update(dict_)
        return final_dict
