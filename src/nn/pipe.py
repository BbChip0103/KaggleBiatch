import gc
import glob
import json
import os
import random
import time
import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import DataParallel

from dataset.dataset import Dataset
from dataset.generator import ItemGenerator
from dataset.sampler import RandomSampler, SequentialSampler
from .classifier import Clf

from utils import utils as utils
from nets.base import Siamese
log = logging.getLogger(__name__)
custom_collate = default_collate


class Pipe(object):
    def __init__(self, config):
        self.config = config

        if self.config['pretrained_weights'] is not None:
            self.test_nn()

    def test_nn(self):
        net = utils.load_class(".".join(["nets", self.config['net_class']]))(config=self.config)
        if self.config['siamese']:
            net = Siamese(one_head=net)
        net = DataParallel(net)
        net.load_state_dict(torch.load(self.config['pretrained_weights'], map_location='cpu')['net'])
        del net
        gc.collect()

    def __call__(self):
        self.ds = Dataset(config=self.config)
        self.ds()
        test_loader_list = self.generate_tta_loader_list(self.ds.test, type="test")
        log.info(f"Processing on folds {self.config['use_folds']}")
        for fold_num in self.config['use_folds']:
            log.info(f"Processing fold {fold_num}.")
            payload = self.generate_loaders(fold_num=fold_num)
            payload.update({"test_loader_list": test_loader_list,
                            "fold_num": fold_num,
                            "loader_names": self.config['augs']['tta']['augs'],
                            "class_weights": self.ds.class_weights[fold_num].to(device=self.config['device'])})
            classifier = Clf(config=self.config, payload=payload)
            if self.config["mode"] in ["train", 'pipe']:
                classifier.train()

            classifier.predict()

    def generate_loaders(self, fold_num):
        log.info(f"AUGs on train {self.config['augs']['train']['augs']} with p = {self.config['augs']['train']['aug_p']} ")
        if self.config['dataset'].get("sampler", False):
            custom_sampler = utils.load_class(".".join(["dataset", self.config['dataset']['sampler']]))
        else:
            custom_sampler = RandomSampler
        log.info(f"Using Sampler: {custom_sampler.__name__}")

        train_data = {k: self.ds.train[k] for k in self.ds.splits['train'][fold_num]}
        train_ds = ItemGenerator(data=train_data,
                                 config=self.config,
                                 local_config={"augs_list": self.config['augs']['train']['augs'],
                                               "aug_p": self.config['augs']['train']['aug_p'],
                                               "include_target": True})
        train_loader = DataLoader(dataset=train_ds,
                                  batch_size=self.config['batch_size'],
                                  collate_fn=custom_collate,
                                  sampler=custom_sampler(data_source=train_data),
                                  num_workers=self.config['n_threds'],
                                  pin_memory=(self.config['device']=="cuda"))
        val_data = {k: self.ds.train[k] for k in self.ds.splits['val'][fold_num]}
        val_ds = ItemGenerator(data=val_data,
                               config=self.config,
                               local_config={"augs_list": [None],
                                             "aug_p": 0,
                                             "include_target": True})
        valid_loader = DataLoader(dataset=val_ds,
                                  batch_size=self.config['batch_size'],
                                  collate_fn=custom_collate,
                                  sampler=SequentialSampler(data_source=val_data),
                                  num_workers=self.config['n_threds'],
                                  pin_memory=(self.config['device']=="cuda"))
        valid_loader_oof_list = self.generate_tta_loader_list(val_data, type='val')
        res = {"train_loader": train_loader,
               "valid_loader": valid_loader,
               "valid_loader_oof_list": valid_loader_oof_list}
        return res

    def generate_tta_loader_list(self, data, type):
        res_list = []
        log.info(f"TTA for {type} on {self.config['augs']['tta']['augs']} augmentations")
        for augs_func in self.config['augs']['tta']['augs']:
            local_config = {"augs_list": [augs_func],
                            "aug_p": 1,
                            "include_target": False}
            ds = ItemGenerator(data=data, config=self.config, local_config=local_config)
            tl = DataLoader(dataset=ds,
                            batch_size=self.config['batch_size'],
                            collate_fn=custom_collate,
                            num_workers=self.config['n_threds'],
                            sampler=SequentialSampler(data_source=data),
                            pin_memory=(self.config['device']=="cuda"))

            res_list.append(tl)
        return res_list
