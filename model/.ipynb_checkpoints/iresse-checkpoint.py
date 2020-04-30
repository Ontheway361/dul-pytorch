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
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False),
                            nn.Sigmoid())

    def forward(self, x):
        return x * self.se_layer(x)


class IR_BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):

        super(IR_BasicBlock, self).__init__()
        self.ir_basic = nn.Sequential(
                            nn.BatchNorm2d(inplanes),
                            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(planes),
                            nn.PReLU(),
                            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                            nn.BatchNorm2d(planes))

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.prelu  = nn.PReLU()
        if self.use_se:
            self.se = SEBlock(planes)


    def forward(self, x):
        
        residual = x
        x = self.ir_basic(x)
        if self.use_se:
            x = self.se(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.prelu(x)
        return x

    
class IR_Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):

        super(IR_Bottleneck, self).__init__()
        self.ir_bottle = nn.Sequential(
                             nn.BatchNorm2d(inplanes),
                             nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                             nn.BatchNorm2d(planes),
                             nn.PReLU(),
                             nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                             nn.BatchNorm2d(planes),
                             nn.PReLU(),
                             nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
                             nn.BatchNorm2d(planes * self.expansion),
                             
                    )
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.prelu  = nn.PReLU()
        if self.use_se:
            self.se = SEBlock(planes * self.expansion)


    def forward(self, x):

        residual = x
        x = self.ir_bottle(x)
        if self.use_se:
            x = self.se(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.prelu(x)
        return x

    
class IResNet(nn.Module):

    def __init__(self, block, layers, feat_dim = 512, drop_ratio = 0.5, use_se = True):

        super(IResNet, self).__init__()

        self.inplanes = 64
        self.use_se = use_se
        self.input_layer = nn.Sequential(
                               nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                               nn.BatchNorm2d(self.inplanes),
                               nn.PReLU(),
                               nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.output_layer = nn.Sequential(
                                nn.BatchNorm2d(512 * block.expansion),
                                nn.Dropout(p=drop_ratio),
                                Flatten(),
                                nn.Linear(512 * block.expansion * 7 * 7, feat_dim),  # size / 16
                                nn.BatchNorm1d(feat_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x


def iresnet_zoo(backbone = 'iresse50', feat_dim = 512, drop_ratio = 0.5, use_se = True):
    
    version_dict = {
        'iresse18' : [2, 2, 2, 2],
        'iresse34' : [3, 4, 6, 3],
        'iresse50' : [3, 4, 14,3],  # default = [3, 4, 6, 3]
        'iresse100': [3, 13,30,3],  # just for face
        'iresse101': [3, 4, 23,3],
        'iresse152': [3, 8, 36,3],
    }
    if backbone == 'iresse18' or backbone == 'iresse34' or backbone == 'iresse50':
        block = IR_BasicBlock
    else:
        block = IR_Bottleneck
        
    return IResNet(block, version_dict[backbone], feat_dim, drop_ratio, use_se)


if __name__ == "__main__":

    model = iresnet_zoo('iresse101', use_se=True)
    input = torch.randn(1, 3, 112, 112)
    flops, params = thop.profile(model, inputs=(input, ))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)

