import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)


# https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score

class EuclidianDistance(nn.Module):
    def __init__(self, p = 2, eps = 1e-16):
        super(EuclidianDistance, self).__init__()
        self.norm = p
        self.eps = eps

    def forward(self, x1, x2):
        num = x1.size(0)
        x1 = x1.view(num, -1)
        x2 = x2.view(num, -1)
        res =  F.pairwise_distance(x1, x2, self.norm, self.eps)
        res = res / (75 * 75 * num)

        res = torch.mean(res)
        return res

#https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def dice_loss(m1, m2):
    num = m1.size(0)
    m1 = m1.view(num, -1)
    m2 = m2.view(num, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = score.sum() / num
    return score


if __name__ == '__main__':
    loss = EuclidianDistance()
    # x = Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))
    # y = Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))

    x = Variable(torch.randn(10, 1, 75, 75))
    y = Variable(torch.randn(10, 1, 75, 75))

    res = loss.forward(x,y)
    print(res)
