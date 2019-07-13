from collections import OrderedDict

from pretrainedmodels.models import senet
import torch.nn as nn

from .base import BaseNet


class SENet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_num = 3
        self.model = senet.senet154(num_classes=1000, pretrained="imagenet")
        self.dropout = nn.Dropout2d(p=0.2)
        self.get_last_linear(in_features=2048)

    def model_features(self, x):
        x = self.model.layer0(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
