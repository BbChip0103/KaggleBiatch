import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/44283

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



class TestNet_one_d(nn.Module):
    def __init__(self, in_shape=(1,16000), num_classes=31 ):
        super(TestNet_one_d, self).__init__()

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

        self.in_channels = 30
        self.scale = SEScale(self.in_channels, self.in_channels // 2)


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
        x = F.dropout(x,p=0.20,training=self.training)
        x = F.relu(self.conv10a(x),inplace=True)
        x = F.relu(self.conv10b(x),inplace=True)
        x = F.max_pool1d(x,kernel_size=2,stride=2)
        #------------------------------------------

        # print(x.size())
        x = F.adaptive_avg_pool1d(x,1)
        # x = self.scale(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,p=0.50,training=self.training)
        x = F.relu(self.linear1(x),inplace=True)

        x = F.dropout(x,p=0.50,training=self.training)
        x = F.relu(self.linear2(x),inplace=True)
        x = self.fc(x)

        return x  #logits

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

if __name__ == '__main__':
    net = TestNet_one_d()
    x = Variable(torch.randn(10, 1, 16000))
    # x =  Variable(torch.randn(1, 2, 2, 20000))
    y = net(x)
    print(y)