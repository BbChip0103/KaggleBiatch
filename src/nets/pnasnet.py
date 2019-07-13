from collections import OrderedDict

from pretrainedmodels.models import pnasnet
import torch.nn as nn

from .base import BaseNet


class PNasNet(BaseNet):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_num = 8
        self.model = pnasnet.pnasnet5large(num_classes=1000, pretrained="imagenet")
        self.dropout = nn.Dropout2d(p=0.2)
        self.get_last_linear(in_features=4320)

    def model_features(self, x):
        x_conv_0 = self.model.conv_0(x)
        x_stem_0 = self.model.cell_stem_0(x_conv_0)
        x_stem_1 = self.model.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.model.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.model.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.model.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.model.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.model.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.model.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.model.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.model.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.model.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.model.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.model.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.model.cell_11(x_cell_9, x_cell_10)
        return x_cell_11
