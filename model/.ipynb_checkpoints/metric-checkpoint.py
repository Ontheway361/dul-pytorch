#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from IPython import embed

class AngleLoss(nn.Module):

    def __init__(self, gamma = 0):

        super(AngleLoss, self).__init__()
        self.it        = 0
        self.LambdaMin = 5.0
        self.gamma     = gamma
        self.LambdaMax = 1500.0
        self.lamb      = 1500.0

        
    def forward(self, input, target):

        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)   # size = (B, 1)
        gt_mat = cos_theta.data * 0.0 # size = (B, Classnum)
        gt_mat.scatter_(1, target.data.view(-1,1), 1)
        gt_mat = gt_mat.byte()
        gt_mat = Variable(gt_mat)
        
        # soft 
        self.lamb = max(self.LambdaMin, self.LambdaMax/(1+0.1*self.it))
        output = cos_theta * 1.0 
        output[gt_mat] -= cos_theta[gt_mat] * 1.0 / (1+self.lamb)
        output[gt_mat] += phi_theta[gt_mat] * 1.0 / (1+self.lamb)

        outprob = F.log_softmax(output, dim=1)
        logprob = outprob.gather(1, target).view(-1)
        prob    = Variable(logprob.data.exp())
        loss = -1 * ((1 - prob)**self.gamma) * logprob
        loss = loss.mean()

        return loss, outprob
