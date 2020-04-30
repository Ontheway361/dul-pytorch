#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from IPython import embed

class Arcface(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, args):

        super(Arcface, self).__init__()

        self.in_features  = args.in_feats
        self.out_features = args.classnum
        self.s = args.scale
        self.m = args.margin
        self.weight = torch.FloatTensor(args.classnum, args.in_feats)
        if args.use_gpu:
            self.weight = self.weight.cuda()
        self.weight = Parameter(self.weight)
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.cos_m = math.cos(args.margin)
        self.sin_m = math.sin(args.margin)
        self.th = math.cos(math.pi - args.margin)
        self.mm = math.sin(math.pi - args.margin) * args.margin


    def forward(self, input, target):

        # cos(a+b) = cos(a)cos(b)-sin(a)sin(b)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class Cosface(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self,  args):

        super(Cosface, self).__init__()

        self.in_features  = args.in_feats
        self.out_features = args.classnum
        self.s = args.scale    # default s = 30
        self.m = args.margin   # default m = 0.4
        self.weight = torch.FloatTensor(out_features, in_features)
        if args.use_gpu:
            self.weight = self.weight.cuda()
        self.weight = Parameter(self.weight)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, target):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        output *= self.s

        return output


class Sphereface(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, args):

        super(Sphereface, self).__init__()

        self.in_features  = args.in_feats
        self.out_features = args.classnum
        self.m     = args.margin   # default : m = 4
        self.base  = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter   = 0
        self.weight = torch.FloatTensor(args.classnum, args.in_feat)
        if args.use_gpu:
            self.weight = self.weight.cuda()
        self.weight = Parameter(self.weight)
        nn.init.xavier_uniform_(self.weight)
        
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, target):

        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        one_hot = torch.zeros(cos_theta.size(), device=input.device)
        one_hot.scatter_(1, target.view(-1, 1), 1)

        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

def metric_zoo(args):

    if args.metric == 'arc':
        return Arcface(args)
    elif args.metric == 'cos':
        return Cosface(args)
    elif args.metric == 'sphere':
        return Sphereface(args)
    else:
        raise TypeError('metric must be arc, add or sphere ...')
