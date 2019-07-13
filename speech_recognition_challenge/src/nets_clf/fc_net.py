import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

class SEScale(nn.Module):
    def __init__(self, in_channels, reduction = 16):
        super(SEScale, self).__init__()
        channel = in_channels
        self.fc1 = nn.Linear(channel, reduction)
        self.fc2 = nn.Linear(reduction, channel)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace = True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

class FC_net(nn.Module):
    def __init__(self):
        super(FC_net, self).__init__()
        self.in_channels = 31 * 10
        self.num_classes = 31

        self.scale = SEScale(self.in_channels, self.in_channels // 2)
        self.linear1 = nn.Linear(self.in_channels, 100)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(100, 50)
        self.relu2 = nn.PReLU()
        self.fc = nn.Linear(50, 31)

    def forward(self, x ):
        x = self.scale(x) * x
        # print(x.size())
        # x = x.sum(dim = 1)
        # print(x.size())
        x = self.linear1(x)
        x = self.relu1(x)
        x = F.dropout(x, p = 0.2)
        x = self.linear2(x)
        x = self.relu2(x)
        x = F.dropout(x, p = 0.2)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = FC_net()
    x = Variable(torch.randn(10, 155))

    y = net(x)
    print('output', y.size())