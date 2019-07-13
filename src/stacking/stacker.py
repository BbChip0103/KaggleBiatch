import json
import logging

import numpy as np

from dataset.clf_dataset import ClfDataset
from stacking.classifier import Classifier
import utils.utils as utils
log = logging.getLogger(__name__)


class Stacker(object):
    def __init__(self, config):
        self.config = config
        log.info(f"Stacking folders {config['mode_stack']['folders']}")
        self.clf_ds = ClfDataset(config=self.config)

    def __call__(self):
        self.clf_ds()
        self.clf = Classifier(config=self.config, data=self.clf_ds.data)
        self.clf()
