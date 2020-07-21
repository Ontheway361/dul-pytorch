#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed


class FaceLoss(nn.Module):
    ''' Classic loss function for face recognition '''

    def __init__(self, args):

        super(FaceLoss, self).__init__()
        self.args     = args


    def forward(self, predy, target, mu = None, logvar = None):

        loss = None
        if self.args.loss_mode == 'focal_loss':
            logp = F.cross_entropy(predy, target, reduce=False)
            prob = torch.exp(-logp)
            loss = ((1-prob) ** self.args.loss_power * logp).mean()

        elif self.args.loss_mode == 'hardmining':
            batchsize = predy.shape[0]
            logp      = F.cross_entropy(predy, label, reduce=False)
            inv_index = torch.argsort(-logp) # from big to small
            num_hard  = int(self.args.hard_ratio * batch_size)
            hard_idx  = ind_sorted[:num_hard]
            loss      = torch.sum(F.cross_entropy(pred[hard_idx], label[hard_idx]))

        else: # navie-softmax
            loss = F.cross_entropy(predy, target)

        if (mu is not None) and (var is not None):
            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            kl_loss = kl_loss.sum(dim=1).mean()
            loss    = loss + self.args.kl_tdoff * kl_loss
        return loss
