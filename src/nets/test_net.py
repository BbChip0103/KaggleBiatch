import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet


class TestNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_num = 1
        self.model = nn.Sequential(nn.Conv2d(3,
                                             out_channels=32,
                                             kernel_size=3,
                                             padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2)
                                   )

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=0.2)
        self.get_last_linear(in_features=64)

    def model_features(self, x):
        x = self.model(x)
        x = self.pool(F.relu(self.conv2(x)))
        return x
