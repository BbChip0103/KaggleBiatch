import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LastLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.out_features = out_features
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.register_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * 1, grad_i[1] * 1))

    def reset(self):
        self.register_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * 1, grad_i[1] * 1))

    def forward(self, x):
        x = self.layer(x)
        return x


class BaseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.config['competition_num_channels'] != 3:
            self.first_layer = nn.Conv2d(self.config['competition_num_channels'], 3, kernel_size=1)

    def get_last_linear(self, in_features):
        self.last_linear = LastLinear(in_features=in_features, out_features=1)#self.config['num_classes'])

    def features(self, x):
        if self.config['competition_num_channels'] != 3:
            x = self.first_layer(x)
        x = self.model_features(x)
        return x

    def logits(self, features):
        x = nn.ReLU()(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        x = self.last_linear(x)
        if self.last_linear.out_features == 1:
            x = x.squeeze(dim=1)
        return x

    def freeze(self):
        log.info("Freezing net.")
        child_counter = 0
        for child_counter, child in enumerate(self.model.children()):
            if child_counter < self.freeze_num:
                log.info(f"Layer {child.__class__.__name__} {child_counter} was frozen")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                log.info(f"Layer {child.__class__.__name__} {child_counter} was NOT frozen")

    def unfreeze(self):
        log.info("Unfreezing net.")
        ct = 0
        for name, child in self.model.named_children():
            ct += 1
            if ct >= 0:
                for name2, params in child.named_parameters():
                    params.requires_grad = True

    def reset(self):
        self.last_linear.reset()


class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Siamese(nn.Module):
    def __init__(self, one_head):
        super().__init__()
        self.one_head = one_head
        self.one_head.reset()
        n_feats = self.one_head.last_linear.layer.in_features
        self.conv_1d_last = ConvBnRelu1d(n_feats * 4, n_feats)

    def forward(self, x):
        N_crops = x.shape[1]
        heads_out = []
        for i in range(N_crops):
            tmp_x = self.one_head.features(x[:, i, ::])
            tmp_x = self.one_head.logits(tmp_x)
            heads_out.append(tmp_x)#.unsqueeze(dim=1)
        x = torch.cat(heads_out, dim=-1)

        x = self.conv_1d_last(x)
        # x = x.squeeze(dim=1)
        x = self.one_head.last_linear(x)
        return x

    def freeze(self):
        self.one_head.freeze()

    def unfreeze(self):
        self.one_head.unfreeze()

    def reset(self):
        self.one_head.reset()
