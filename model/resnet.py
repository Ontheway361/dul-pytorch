#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import thop
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    
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
    
    
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=False):
        
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.use_se=use_se
        if use_se:
            self.se = SEBlock(planes)

        
    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=False):

        super(Bottleneck, self).__init__()
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.use_se= use_se
        if use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class ResNet(nn.Module):

    def __init__(self, block, layers, feat_dim=512, drop_ratio = 0.5, use_se = False):
        
        super(ResNet, self).__init__()
        
        self.inplanes = 64
        self.use_se   = use_se
        self.feat_dim = feat_dim
        self.layer0 = nn.Sequential(
                          nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.BatchNorm2d(self.inplanes),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.mu_head = nn.Sequential(
                                nn.BatchNorm2d(512 * block.expansion),
                                nn.Dropout(p=drop_ratio),
                                Flatten(),
                                nn.Linear(int(512 * block.expansion * 7 * 7), int(feat_dim)),  # size / 16
                                nn.BatchNorm1d(int(feat_dim)))
        
        self.var_head = nn.Sequential(
                                nn.BatchNorm2d(512 * block.expansion),
                                nn.Dropout(p=drop_ratio),
                                Flatten(),
                                nn.Linear(int(512 * block.expansion * 7 * 7), int(feat_dim)),  # size / 16
                                nn.BatchNorm1d(int(feat_dim)))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


                    
    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        mux = self.mu_head(x)
        var = F.normalize(torch.exp(self.var_head(x)))  # normlize
        if self.training:
             mux = mux + torch.randn(self.feat_dim) * torch.sqrt(var)
        return mux, var


def resnet_zoo(backbone = 'resnet50', feat_dim = 512, drop_ratio = 0.4, use_se = False):
    
    version_dict = {
        'resnet18' : [2, 2, 2, 2],
        'resnet34' : [3, 4, 6, 3],
        'resnet50' : [3, 4, 6, 3],
        'resnet101': [3, 4, 23,3],
        'resnet152': [3, 8, 36,3],
    }
    if backbone == 'resnet18' or backbone == 'resnet34':
        block = BasicBlock
    else:
        block = Bottleneck
        
    return ResNet(block, version_dict[backbone], feat_dim, drop_ratio, use_se)



if __name__ == "__main__":
    
    input = torch.randn(10, 3, 112, 112)
    model = resnet_zoo('resnet18', use_se=True)
    model.eval()
    # flops, params = thop.profile(model, inputs=(input, ))
    # flops, params = thop.clever_format([flops, params], "%.3f")
    # print(flops, params)
    mu, sig = model(input)
    embed()