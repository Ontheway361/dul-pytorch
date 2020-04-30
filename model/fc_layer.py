#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed


class FullyConnectedLayer(nn.Module):
    
    def __init__(self, args):
        
        super(FullyConnectedLayer, self).__init__()
        
        self.args   = args
        self.weight = nn.Parameter(torch.Tensor(args.classnum, args.in_feats))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m  = math.cos(args.margin)
        self.sin_m  = math.sin(args.margin)
        self.mm     = math.sin(math.pi - self.args.margin) * self.args.margin
        self.register_buffer('factor_t', torch.zeros(1))
        self.iter  = 0
        self.base  = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        
    def forward(self, x, label):  
        
        cos_theta  = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta  = cos_theta.clamp(-1, 1) 
        batch_size = label.size(0)
        cosin_simi = cos_theta[torch.arange(0, batch_size), label].view(-1, 1) 

        if self.args.fc_mode == 'softmax':
            score = cosin_simi
            
        elif self.args.fc_mode == 'sphereface':
            self.iter  += 1
            self.lamb   = max(self.lambda_min, self.base * (1 + self.alpha * self.iter) ** (-1 * self.power))
            cos_theta_m = self.mlambda[int(self.args.margin)](cosin_simi) 
            theta       = cosin_simi.data.acos()
            k           = ((self.args.margin * theta) / math.pi).floor()
            phi_theta   = ((-1.0) ** k) * cos_theta_m - 2 * k
            score       = (self.lamb * cosin_simi + phi_theta) / (1 + self.lamb)
            
        elif self.args.fc_mode == 'cosface':
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cosin_simi - self.args.margin, cosin_simi)
            else:
                score = cosin_simi - self.args.margin
                
        elif self.args.fc_mode == 'arcface':
            sin_theta   = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m  
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
            else:
                score = cos_theta_m
                
        elif self.args.fc_mode == 'mvcos':
            mask        = cos_theta > cosin_simi - self.args.margin
            hard_vector = cos_theta[mask]
            if self.args.hard_mode == 'adaptive':
                cos_theta[mask] = (self.args.t + 1.0) * hard_vector + self.args.t  # Adaptive
            else:
                cos_theta[mask] = hard_vector + self.args.t  # Fixed
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cosin_simi - self.args.margin, cosin_simi)
            else:
                score = cosin_simi - self.args.margin
                
        elif self.args.fc_mode == 'mvarc':
            sin_theta   = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            mask        = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            if self.args.hard_mode == 'adaptive':
                cos_theta[mask] = (self.args.t + 1.0) * hard_vector + self.args.t  # Adaptive
            else:
                cos_theta[mask] = hard_vector + self.args.t # Fixed
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
            else:
                score = cos_theta_m
        
        elif self.args.fc_mode == 'curface':
            with torch.no_grad():
                origin_cos = cos_theta
            sin_theta   = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            mask        = cos_theta > cos_theta_m
            score       = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi - self.mm)
            hard_sample = cos_theta[mask]
            with torch.no_grad():
                self.factor_t = cos_theta_m.mean() * 0.01 + 0.99 * self.factor_t
            cos_theta[mask] = hard_sample * (self.factor_t + hard_sample)
        else:
            raise Exception('unknown fc type!')

        cos_theta.scatter_(1, label.data.view(-1, 1), score)
        cos_theta *= self.args.scale
        return cos_theta
