import torch.nn as nn
from pretrainedmodels.models import inceptionresnetv2

from .base import BaseNet


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0.1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class IncepResV2(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_num = 8
        self.model = inceptionresnetv2(num_classes=1000, pretrained="imagenet")
        self.dropout = nn.Dropout2d(p=0.2)
        self.get_last_linear(in_features=1536)

    def model_features(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.maxpool_5a(x)
        x = self.model.mixed_5b(x)
        x = self.model.repeat(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_2(x)
        x = self.model.block8(x)
        x = self.model.conv2d_7b(x)
        return x
