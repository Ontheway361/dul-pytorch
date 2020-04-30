#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import thop
import math
import torch
from torch import nn
from IPython import embed
from model.rescbam import ChannelAttention, SpatialAttention

class BottleNeck(nn.Module):
    def __init__(self, inp, oup, stride, expansion, use_cbam = False):
        
        super(BottleNeck, self).__init__()
        
        self.connect = stride == 1 and inp == oup
        self.use_cbam = use_cbam
        self.conv = nn.Sequential(
            # 1*1 conv
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 3*3 depth wise conv
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 1*1 conv
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup))
        if use_cbam:
            self.ca = ChannelAttention(oup)
            self.sa = SpatialAttention()
       

    def forward(self, x):

        out = self.conv(x)
        if self.use_cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out
        if self.connect:
            return x + out
        else:
            return out


class ConvBlock(nn.Module):
    
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)

        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MobileFace(nn.Module):
    def __init__(self, feat_dim = 512, drop_ratio = 0.5, use_cbam = False):
        
        super(MobileFace, self).__init__()
        
        self.conv1    = ConvBlock(3, 64, 3, 2, 1)
        self.dwconv1  = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.use_cbam = use_cbam
        self.cur_channel = 64 #t, c,  n,  s
        self.block_setting = [[2, 64, 5,  2],
                              [4, 128, 1, 2],
                              [2, 128, 6, 1],
                              [4, 128, 1, 2],
                              [2, 128, 2, 1]]
        self.layers = self._make_layer()

        self.conv2   = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)   # CORE
        self.linear1 = ConvBlock(512, feat_dim, 1, 1, 0, linear=True)
        
        # Arcface : BN-Dropout-FC-BN to get the final 512-D embedding feature
        '''
        self.output_layer = nn.Sequential(
                                nn.BatchNorm2d(512),
                                nn.Dropout(p=drop_ratio),
                                Flatten(),
                                nn.Linear(512 * 7 * 7, feat_dim),  # size / 16
                                nn.BatchNorm1d(feat_dim))
        '''
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                
    def _make_layer(self, block = BottleNeck):
        layers = []
        for t, c, n, s in self.block_setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t, self.use_cbam))
                else:
                    layers.append(block(self.cur_channel, c, 1, t, self.use_cbam))
                self.cur_channel = c

        return nn.Sequential(*layers)

    
    def forward(self, x):

        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.layers(x)
        x = self.conv2(x)
        # x = self.output_layer(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    input = torch.Tensor(1, 3, 112, 112)
    model = MobileFace(use_cbam=True)
    flops, params = thop.profile(model, inputs=(input, ))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)
    # model = model.eval()
#     out = model(input)
#     print(out.shape)

