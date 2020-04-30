#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from IPython import embed


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    '''Channel Attention Module'''
    
    def __init__(self, in_planes, reduction = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.max_pool   = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
                                nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        weights  = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        return self.sigmoid(weights)

    
class SpatialAttention(nn.Module):
    ''' Spatial Attention Module '''
    
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        weights    = self.conv1(torch.cat([avg_out, max_out], dim=1)) 
        return self.sigmoid(weights)

    
class CBAM_BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.ca    = ChannelAttention(planes)
        self.sa    = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SAM_BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.sa    = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
    
class CBAM_Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class ResNet(nn.Module):

    def __init__(self, block, layers, feat_dim = 512, drop_ratio = 0.5):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
                               nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                               nn.BatchNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x


def rescbam_zoo(backbone = 'resnet50_cbam', feat_dim = 512, drop_ratio = 0.5):
    
    version_dict = {
        'resnet18_cbam' : [2, 2, 2, 2],
        'resnet34_cbam' : [3, 4, 6, 3],
        'resnet50_cbam' : [3, 4, 14,3],   # default = [3, 4, 6, 3]
        'resnet101_cbam': [3, 4, 23,3],
        'resnet152_cbam': [3, 8, 36,3],
    }
    if backbone == 'resnet18_cbam' or backbone == 'resnet34_cbam' or backbone == 'resnet50_cbam':
        block = CBAM_BasicBlock   # TODO
    else:
        block = CBAM_Bottleneck
        
    return ResNet(block, version_dict[backbone], feat_dim, drop_ratio)



if __name__ == "__main__":
    net = rescbam_zoo('resnet50_cbam')
    print(net)
    x = net(torch.randn(2, 3, 112, 112))
    print(x.shape)