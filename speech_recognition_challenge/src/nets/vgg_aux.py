import torch.nn as nn
import os
import torch
from functools import reduce
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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


class ConvBn1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm1d(out_channels)

        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Simple1dNet(nn.Module):
    def __init__(self, in_shape=(1,16000), num_classes=31 ):
        super(Simple1dNet, self).__init__()

        self.conv1a = ConvBn1d(  1,  8, kernel_size=3, stride=1)
        self.conv1b = ConvBn1d(  8,  8, kernel_size=3, stride=1)

        self.conv2a = ConvBn1d(  8, 16, kernel_size=3, stride=1)
        self.conv2b = ConvBn1d( 16, 16, kernel_size=3, stride=1)

        self.conv3a = ConvBn1d( 16, 32, kernel_size=3, stride=1)
        self.conv3b = ConvBn1d( 32, 32, kernel_size=3, stride=1)

        self.conv4a = ConvBn1d( 32, 64, kernel_size=3, stride=1)
        self.conv4b = ConvBn1d( 64, 64, kernel_size=3, stride=1)

        self.conv5a = ConvBn1d( 64,128, kernel_size=3, stride=1)
        self.conv5b = ConvBn1d(128,128, kernel_size=3, stride=1)

        self.conv6a = ConvBn1d(128,256, kernel_size=3, stride=1)
        self.conv6b = ConvBn1d(256,256, kernel_size=3, stride=1)

        self.conv7a = ConvBn1d(256,256, kernel_size=3, stride=1)
        self.conv7b = ConvBn1d(256,256, kernel_size=3, stride=1)

        self.conv8a = ConvBn1d(256,512, kernel_size=3, stride=1)
        self.conv8b = ConvBn1d(512,512, kernel_size=3, stride=1)

        self.conv9a = ConvBn1d(512,512, kernel_size=3, stride=1)
        self.conv9b = ConvBn1d(512,512, kernel_size=3, stride=1)
        #self.linear1 = nn.Linear(512*31,1024)

        self.conv10a = ConvBn1d( 512,1024, kernel_size=3, stride=1)
        self.conv10b = ConvBn1d(1024,1024, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        self.fc      = nn.Linear(256,num_classes)



    def forward(self, x):

        #print(x.size())
        x = F.relu(self.conv1a(x),inplace=True)
        x = F.relu(self.conv1b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        #x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv2a(x),inplace=True)
        x = F.relu(self.conv2b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        #x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv3a(x),inplace=True)
        x = F.relu(self.conv3b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv4a(x),inplace=True)
        x = F.relu(self.conv4b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.10,training=self.training)
        x = F.relu(self.conv5a(x),inplace=True)
        x = F.relu(self.conv5b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv6a(x),inplace=True)
        x = F.relu(self.conv6b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv7a(x),inplace=True)
        x = F.relu(self.conv7b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv8a(x),inplace=True)
        x = F.relu(self.conv8b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv9a(x),inplace=True)
        x = F.relu(self.conv9b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)

        #print(x.size())
        # x = F.dropout(x,p=0.20,training=self.training)
        # x = F.relu(self.conv10a(x),inplace=True)
        # x = F.relu(self.conv10b(x),inplace=True)
        # x = F.max_pool1d(x,kernel_size=2,stride=2)
        #------------------------------------------

        #print(x.size())
        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)
        # x = F.dropout(x,p=0.50,training=self.training)
        # x = F.relu(self.linear1(x),inplace=True)

        # x = F.dropout(x,p=0.50,training=self.training)
        # x = F.relu(self.linear2(x),inplace=True)
        # x = self.fc(x)

        return x  #logits
    

class VGG19bn_aux(nn.Module):
    def __init__(self):
        super(VGG19bn_aux, self).__init__()
        self.num_classes = 31#12
        self.model =  VGG(make_layers(cfg['E'], batch_norm=True))

        model_list = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)] + list(self.model.features.children())[1:]
        self.model.features = nn.Sequential(*model_list)

        self.avgpool = nn.AvgPool2d(3)
        self.last_linear = nn.Linear(512, self.num_classes)

        self.aux_net = Simple1dNet()

        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        self.fc      = nn.Linear(256,self.num_classes)

    def forward(self, img, aux):
        x = img
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        y = self.aux_net(aux)

        x = torch.cat([x,y], 1)


        x = F.dropout(x,p=0.50,training=self.training)
        x = F.relu(self.linear1(x),inplace=True)

        x = F.dropout(x,p=0.50,training=self.training)
        x = F.relu(self.linear2(x),inplace=True)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    test_flag = True
    net = VGG19bn_aux()
    x = Variable(torch.randn(10, 1, 161, 99))
    a = Variable(torch.randn(10, 1, 16000))
    y = net(*[x,a])
    print('output', y.size())