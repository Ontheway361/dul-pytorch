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
        return input.view(input.size(0), -1)


class SEBlock(nn.Module):
    ''' Squeeze and Excitation Module '''
    def __init__(self, channels, reduction = 16):

        super(SEBlock, self).__init__()
        self.se_layer = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
                            nn.PReLU(),
                            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False),
                            nn.Sigmoid())

    def forward(self, x):
        return x * self.se_layer(x)


class IR_BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):

        super(IR_BasicBlock, self).__init__()
        self.ir_basic = nn.Sequential(
                            nn.BatchNorm2d(inplanes, eps=2e-5),
                            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(planes, eps=2e-5),
                            nn.PReLU(),
                            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                            nn.BatchNorm2d(planes, eps=2e-5))

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.prelu  = nn.PReLU()
        if self.use_se:
            self.se = SEBlock(planes)


    def forward(self, x):
        residual = x
        out = self.ir_basic(x)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # x = self.prelu(x)  # TODO
        return out + residual


class IR_Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IR_Bottleneck, self).__init__()
        self.ir_bottle = nn.Sequential(
                             nn.BatchNorm2d(inplanes, eps=2e-5),
                             nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                             nn.BatchNorm2d(planes, eps=2e-5),
                             nn.PReLU(),
                             nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                             nn.BatchNorm2d(planes, eps=2e-5),
                             nn.PReLU(),
                             nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
                             nn.BatchNorm2d(planes * self.expansion, eps=2e-5))

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.prelu  = nn.PReLU()
        if self.use_se:
            self.se = SEBlock(planes * self.expansion)


    def forward(self, x):
        residual = x
        out = self.ir_bottle(x)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
    
        return self.prelu(out + residual)


class DULResNet(nn.Module):
    def __init__(self, block, layers, feat_dim = 512, drop_ratio = 0.4, use_se = True, used_as = 'baseline'):
        '''
        used_as = baseline : just use mu_head for deterministic model
        used_as = dul_cls  : use both mu_head and logvar_head for dul_cls model
        used_as = backbone : neither mu_head nor logvar_head is used, just for feature extracting.
        '''
        super(DULResNet, self).__init__()

        self.inplanes = 64
        self.use_se   = use_se
        self.used_as  = used_as
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.mu_head = nn.Sequential(
            nn.BatchNorm2d(512 * block.expansion, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            nn.Linear(512 * block.expansion * 7 * 7, feat_dim),
            nn.BatchNorm1d(feat_dim, eps=2e-5))

        # use logvar instead of var !!!
        if used_as == 'dul_cls':
            self.logvar_head = nn.Sequential(
                nn.BatchNorm2d(512 * block.expansion, eps=2e-5, affine=False),
                nn.Dropout(p=drop_ratio),
                Flatten(),
                nn.Linear(512 * block.expansion * 7 * 7, feat_dim),
                nn.BatchNorm1d(feat_dim, eps=2e-5))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)


    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.used_as == 'backbone':
            mu = x
            logvar = None, 
            embedding = None
        elif self.used_as == 'baseline':
            mu = None
            logvar = None, 
            embedding = self.mu_head(x)
        else:
            mu = self.mu_head(x)
            logvar = self.logvar_head(x)
            embedding = self._reparameterize(mu, logvar)
        return (mu, logvar, embedding)

    
def dulres_zoo(backbone = 'dulres18', feat_dim = 512, drop_ratio = 0.4, \
               use_se = True, used_as = 'baseline'):
    zoo_dict = {
        'dulres18' : [2, 2, 2, 2],
        'dulres50' : [3, 4, 6, 3],
        'dulres101': [3, 4, 23,3],
    }

    block = IR_BasicBlock if backbone == 'dulres18' else IR_Bottleneck
    return DULResNet(block, zoo_dict[backbone], feat_dim, drop_ratio, use_se, used_as)



if __name__ == "__main__":

    model = dulres_zoo(backbone='dulres18', used_as='baseline')
    input = torch.randn(1, 3, 112, 112)
    flops, params = thop.profile(model, inputs=(input, ))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)
