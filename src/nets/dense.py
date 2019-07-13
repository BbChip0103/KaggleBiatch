from torchvision.models import densenet
import torch.nn as nn

from .base import BaseNet


class DenseNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_num = 2
        feats = densenet.densenet161(num_classes=1000, pretrained="imagenet").children().__next__()
        self.model = nn.Sequential(*list(feats.children())[:-2])
        self.dropout = nn.Dropout2d(p=0.2)
        self.get_last_linear(in_features=1056)

    def model_features(self, x):
        x = self.model(x)
        return x
