import os

from torch.utils.data import DataLoader
import numpy as np

from dataset import dataset
from dataset import generator
from dataset.sampler import RandomSampler
from utils import utils as utils


class Debugger(object):
    def dataset(self, config):
        self.config = config
        local_config = {"augs_list": ["GaussNoise"],
                        "aug_p": 1}
        batch_size = 2
        ds = dataset.Dataset(config=config)
        ds()
        for i, df in enumerate([ds.train, ds.test]):
            local_config["include_target"] = True if i == 0 else False
            ds = generator.ItemGenerator(df, config, local_config)

            if self.config['dataset'].get("sampler", False):
                custom_sampler = utils.load_class(".".join(["dataset", self.config['dataset']['sampler']]))
            else:
                custom_sampler = RandomSampler
            train_loader = DataLoader(ds,
                                      batch_size=batch_size,
                                      sampler=custom_sampler(data_source=df),
                                      num_workers=0,
                                      pin_memory=False)

            for i, dict_ in enumerate(train_loader):
                print("Max", dict_['img'].min().cpu().numpy(),
                      "Min", dict_['img'].max().cpu().numpy(),
                      "Mean", dict_['img'].mean().cpu().numpy(),
                      "Target", dict_.get('target', "No target"),
                      "Shape", dict_['img'].shape,
                      dict_['id'])
                save_path = os.path.join(self.config['out_folder'], "img.npy")
                np.save(save_path, dict_['img'].cpu().numpy())
                if i > 5:
                    break
