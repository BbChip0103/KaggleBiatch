import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.batch = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(61440, 11)

    def forward(self, x):
        x = self.batch(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return x


if __name__ == '__main__':
    net = TestNet()
    x = Variable(torch.randn(1, 1, 161, 99))
    # x =  Variable(torch.randn(1, 2, 2, 20000))
    y = net(x)
    print(y)