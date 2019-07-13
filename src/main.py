import os
import argparse
import json
import logging
import subprocess as sp
from shutil import copyfile
from pathlib import Path
import logging
from multiprocessing import cpu_count

import torch
import torch.multiprocessing

from utils import utils
from nn.pipe import Pipe
from debugging.debug import Debugger
from stacking.stacker import Stacker


torch.multiprocessing.set_sharing_strategy('file_system')


MODES = ["train", "predict", "pipe",
        "debug_dataset",
        "stack", "debug_stack"]


class Main(object):
    def __init__(self):
        self.debugger = Debugger()
        self.get_args()
        self.generate_config()
        self.prepare_infrastructure()
        self.set_logger()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", help="Path to config file", default="configs/base.json")
        parser.add_argument("--mode", help="Mode: train or predict or pipe.", default="pipe")
        parser.add_argument("--target", help="Target: if None -> all, else only one for binary.", default=None)

        args = parser.parse_args()
        self.config = json.load(open(os.path.join(os.getcwd(), args.config), "r"))
        self.config['mode'] = args.mode
        self.config['config_name'] = args.config
        self.config['mode_target'] = args.target

    def is_debug(self):
        if "debug" in self.config['mode']:
            return True
        if "base" in self.config['config_name']:
            return True
        return False

    def get_folder_type(self):
        if self.config['mode'] in ['train', 'predict', 'pipe']:
            res = "nn"
        elif self.is_debug():
            res = "_debug"
        elif "stack" in self.config['mode']:
            res = "stack"
        if self.config['mode_target'] is not None:
            res = os.path.join(res, f"target_{self.config['mode_target']}")
        return res

    def generate_config(self):
        env_file = json.load(open(os.path.join(str(Path.home()), ".kaggle/path.json"), "r"))
        competition_file = json.load(open(os.path.join(os.getcwd(), "competition.json"), "r"))
        aug_json = json.load(open(os.path.join(os.getcwd(), "configs/service/aug.json"), "r"))
        self.config.update(competition_file)
        self.config['augs'] = aug_json
        self.config['dataset'].update(competition_file['dataset_split'])
        self.config['siamese'] = self.config.get("siamese", False)

        if self.config['siamese'] & self.config['dataset']['resize'][0] >= 512:
            self.config["competition_data_folder"] = "protein"
            self.config["competition_img"]["type"] = {"test": "tif", "train": "tif"}
        elif self.config['competition_data_folder'] == "protein/protein_1024":
            self.config["competition_img"]["type"] = {"test": "jpg", "train": "jpg"}
        elif self.config['competition_data_folder'] == "protein/protein_512":
            self.config["competition_img"]["type"] = {"test": "png", "train": "png"}
        else:
            raise NotImplementedError("No such data folder")

        self.config['n_threds'] = cpu_count()
        self.config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        self.config["net_name"] = self.config['net_class'].split(".")[-1]
        self.config['out_path'] = os.path.join(env_file['output_path'],
                                                 self.config['competition_name'])
        self.config['out_folder'] = os.path.join(self.config['out_path'],
                                                 self.get_folder_type())
        self.config['out_folder'], counter = utils.create_path(self.config['out_folder'])
        self.config['visdom_env_name'] = self.config['net_name'] + f"_{counter}"
        self.config['data_folder'] = os.path.join(env_file['data_path'],
                                                  self.config['competition_data_folder'])
        self.config['predictions_folder'] = os.path.join(self.config['out_folder'],
                                                        "predicitons")
        self.config['weights_folder'] =os.path.join(self.config['out_folder'],
                                                    "weights")
        self.config['splits_path'] = os.path.join(str(Path.home()),
                                                  ".kaggle_splits",
                                                  self.config['competition_name'],
                                                  "debug" if self.is_debug() else "",)
        if self.config['pretrained_weights'] is not None:
            self.config['pretrained_weights'] = os.path.join(os.path.split(self.config['out_folder'])[0],
                                                             self.config['pretrained_weights'])
        if self.config['use_folds'] == "all":
            self.config['use_folds'] = list(range(self.config['dataset']['n_folds']))
        self.config['debug'] = False
        if self.is_debug():
            self.config['n_epochs'] = 2
            self.config['mode_train']['unfreeze'] = 1
            self.config['debug'] = True
            # self.config['mode_stack']['early_stopping_rounds'] = 5
            # self.config['mode_stack']["num_rounds"] = 10

        assert len(self.config['mode_train']['loss']) <= 1, "Cannot be more than 1 loss type"
        assert self.config['competition_type'] in ['binary', 'multilabel', 'multiclass', 'segmentation']

    def prepare_infrastructure(self):
        os.makedirs(self.config['out_folder'], exist_ok=True)
        os.makedirs(self.config['splits_path'], exist_ok=True)
        os.makedirs(self.config['predictions_folder'], exist_ok=True)
        os.makedirs(self.config['weights_folder'], exist_ok=True)
        utils.save_src_to_zip(save_path=self.config['out_folder'])

        with open(os.path.join(self.config['out_folder'], "settings.json"), "w") as f:
            json.dump(self.config, f, indent=2, sort_keys=True)

    def set_logger(self):
        self.log = utils.set_logger(out_folder=self.config['out_folder'], name="log")
        self.log.info(f"{self.config}")

    def __call__(self):
        if self.config['mode'] not in MODES:
            raise NotImplementedError("No such mode!")
        else:
            if self.config['mode'] == "debug_dataset":
                self.debugger.dataset(config=self.config)
            elif "stack" in self.config['mode']:
                Stacker(config=self.config)()
            elif self.config['mode'] in ["train", "predict", "pipe"]:
                Pipe(config=self.config)()

        self.log.info(f"Saved to {self.config['out_folder']}")


if __name__ == "__main__":
    Main()()
