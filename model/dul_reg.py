#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class Flatten(nn.Module):
    def forward(self, input):
        """
        Returns the forward input of the input.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return input.view(input.size(0), -1)
    
    
class RegHead(nn.Module):
    
    expansion = 1  # ResNet18
    resnet_out_channels = 512
    
    def __init__(self, feat_dim = 512, drop_ratio = 0.4):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            feat_dim: (int): write your description
            drop_ratio: (float): write your description
        """
        
        super(RegHead, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1) * 1e-4)
        self.beta  = nn.Parameter(torch.ones(1) * (-7))
        self.mu_head = nn.Sequential(
            nn.BatchNorm2d(self.resnet_out_channels, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            nn.Linear(self.resnet_out_channels * self.expansion * 7 * 7, feat_dim),
            nn.BatchNorm1d(feat_dim, eps=2e-5))

        # use logvar instead of var !!!
        self.logvar_head = nn.Sequential(
            nn.BatchNorm2d(self.resnet_out_channels * self.expansion, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            nn.Linear(self.resnet_out_channels * self.expansion * 7 * 7, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim, eps=0.001,  affine=False))
    
    def forward(self, layer4_out):    
        """
        Parameters ---------- layer4 computation

        Args:
            self: (todo): write your description
            layer4_out: (todo): write your description
        """
        mu     = self.mu_head(layer4_out)
        logvar = self.logvar_head(layer4_out)
        logvar = self.gamma * logvar + self.beta
        return (mu, logvar)

