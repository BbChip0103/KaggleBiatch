import torch.nn as nn
from torchvision import models

from .base import BaseNet


class _ResnetBaseNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_num = 5
        self.dropout = nn.Dropout2d(p=0.2)

    def init_last_linear(self):
        self.get_last_linear(in_features=self.model.fc.in_features)

    def model_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Resnet18(_ResnetBaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet.resnet18(pretrained=True)
        self.init_last_linear()


class Resnet34(_ResnetBaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet.resnet34(pretrained=True)
        self.init_last_linear()


class Resnet50(_ResnetBaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet.resnet50(pretrained=True)
        self.init_last_linear()


class Resnet101(_ResnetBaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet.resnet101(pretrained=True)
        self.init_last_linear()


class Resnet152(_ResnetBaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet.resnet152(pretrained=True)
        self.init_last_linear()
