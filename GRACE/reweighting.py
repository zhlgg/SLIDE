import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sd(x):
    return np.std(x, axis=0, ddof=1)


def sd_gpu(x):
    return torch.std(x, dim=0)


def normalize_gpu(x):
    x = F.normalize(x, p=1, dim=1)
    return x


def normalize(x):
    mean = np.mean(x, axis=0)
    std = sd(x)
    std[std == 0] = 1
    x = (x - mean) / std
    return x


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


def lossc(inputs, target, weight):
    loss = nn.NLLLoss(reduce=False)
    return loss(inputs, target).view(1, -1).mm(weight).view(1)


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res


def lossb_expect(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda()
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]

        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss


def lossq(cfeatures, cfs):
    return - cfeatures.pow(2).sum(1).mean(0).view(1) / cfs


def lossn(cfeatures):
    return cfeatures.mean(0).pow(2).mean(0).view(1)


if __name__ == '__main__':
    pass


import torch
import torch.nn as nn
from torch.autograd import Variable

import math



lrbl = 0.25  # learning rate of balance  # 1.0
epochb = 10  # number of epochs to balance  # 20       50
epochs = 30  # number of total epochs to run  # 30
num_f = 1  # number of fourier spaces  # 1
decay_pow = 2 # value of pow for weight decay  # 2
presave_ratio = 0.9  # the ratio for presaving features  # 0.9
lambdap_constant = 0.01  # weight decay for weight1  # 70.0  1.0     0.5
lambda_decay_rate = 0.9  # ratio of epoch for lambda to decay  # 1
min_lambda_times = 0.01  # number of global table levels  # 0.01
lambda_decay_epoch = 5  # number of epoch for lambda to decay  # 5
first_step_cons = 1  # constrain the weight at the first step  # 1

print(lrbl, epochb, epochs, num_f, decay_pow, presave_ratio, lambdap_constant, lambda_decay_rate, min_lambda_times, lambda_decay_epoch, first_step_cons)

"""
cora:
2 20 30 1 2 0.9 0.01 0.975 0.01 5 1
citeseer:
1 20 30 1 2 0.9 0.01 3 0.01 5 1
pubmed:
0.9 20 30 1 2 0.9 0.01 0.975 0.01 5 1
photo:
0.075 30 30 1 2 0.9 0.01 0.9 0.01 5 1
computer:
0.25 20 30 1 2 0.9 0.01 1 0.01 5 1
"""

def lr_setter(optimizer, epoch, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if bl:
        lr = lrbl * (0.1 ** (epoch // (epochb * 0.5)))
    else:
        lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / epochs))) / 1.01)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_learner(cfeatures, pre_features, pre_weight1, global_epoch=0, iter=0, mul_number=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=1.0, momentum=0.9)

    for epoch in range(epochb):
        lr_setter(optimizerbl, epoch, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = lossb_expect(all_feature, softmax(all_weight), num_f, True)
        lossp = softmax(weight).pow(decay_pow).sum()
        lambdap = lambdap_constant * max((lambda_decay_rate ** (global_epoch // lambda_decay_epoch)),
                                     min_lambda_times)
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * presave_ratio + cfeatures * (
                    1 - presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * presave_ratio + weight * (
                    1 - presave_ratio)

    else:
        pre_features = pre_features * presave_ratio + cfeatures * (1 - presave_ratio)
        pre_weight1 = pre_weight1 * presave_ratio + weight * (1 - presave_ratio)

    softmax_weight = softmax(weight) * mul_number

    del weight, cfeaturec, all_feature, all_weight, lossb, lossp, lossg

    return softmax_weight, pre_features, pre_weight1
