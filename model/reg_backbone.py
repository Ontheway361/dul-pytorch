#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import .dul_resnet as dullib
from IPython import embed

class DULReg(dullib.DULResNet):

    def __init__(self, block, layers, feat_dim = 512, drop_ratio = 0.4, use_se = True):

        dullib.DULResNet.__init__(self, block, layers, feat_dim, drop_last, use_se)

        self.mu_head = nn.Sequential(
            nn.BatchNorm2d(512 * block.expansion, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            nn.Linear(512 * block.expansion * 7 * 7, feat_dim),
            nn.BatchNorm1d(feat_dim, eps=2e-5))

        # use logvar instead of var !!!
        self.logvar_head = nn.Sequential(
            nn.BatchNorm2d(512 * block.expansion, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            nn.Linear(512 * block.expansion * 7 * 7, feat_dim),
            nn.BatchNorm1d(feat_dim, eps=2e-5))


    def _loadding_weights(self):
        pass

        
    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        embedding = self._reparameterize(mu, logvar)
        return (mu, logvar, embedding)


def dulres_zoo(backbone = 'dulres18', feat_dim = 512, drop_ratio = 0.4, use_se = True):

    zoo_dict = {
        'dulres18' : [2, 2, 2, 2],
        'dulres50' : [3, 4, 6, 3],
        'dulres101': [3, 4, 23,3],
    }
    if backbone == 'dulres18':
        block = IR_BasicBlock
    else:
        block = IR_Bottleneck
    return DULResNet(block, zoo_dict[backbone], feat_dim, drop_ratio, use_se)


if __name__ == "__main__":

    model = dulres_zoo('dulres18', use_se=True)
    input = torch.randn(1, 3, 112, 112)
    flops, params = thop.profile(model, inputs=(input, ))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)
