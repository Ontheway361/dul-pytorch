#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model as mlib
import dataset as dlib
from config import reg_args

torch.backends.cudnn.bencmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" # TODO

from IPython import embed


class DulRegTrainer(mlib.Faster1v1):

    def __init__(self, args):
        """
        Initialize the faster.

        Args:
            self: (todo): write your description
        """

        mlib.Faster1v1.__init__(self, args)
        self.args    = args
        self.model   = dict()
        self.data    = dict()
        self.result  = dict()
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_gpu = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python    : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.use_gpu))
        print("- IS_DEBUG  : {}".format(self.args.is_debug))
        print('-' * 52)


    def _model_loader(self):
        """
        Run the model.

        Args:
            self: (todo): write your description
        """

        self.model['backbone']  = mlib.dulres_zoo(self.args.backbone, \
                                                  drop_ratio=self.args.drop_ratio, \
                                                  use_se=self.args.use_se,\
                                                  used_as='backbone')  # ResBlock
        self.model['reg_head']  = mlib.RegHead(drop_ratio=self.args.drop_ratio)
        self.model['criterion'] = mlib.RegLoss(feat_dim=self.args.in_feats, classnum=self.args.classnum)
        self.model['optimizer'] = torch.optim.SGD(
                                      [{'params': self.model['reg_head'].parameters()}],
                                       lr=self.args.base_lr,
                                       weight_decay=self.args.weight_decay,
                                       momentum=0.9,
                                       nesterov=True)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], milestones=self.args.lr_adjust, gamma=self.args.gamma)
        if self.use_gpu:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['criterion'] = self.model['criterion'].cuda()

        if self.use_gpu and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            print('Parallel mode was going ...')
        elif self.use_gpu:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['criterion'].fetch_center_from_fc_layer(checkpoint['fc_layer'])
            self.model['backbone'].eval() # backbone was fixed
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')

    
    @staticmethod
    def collate_fn_1v1(batch):
        """
        Collate multiple images into multiple images.

        Args:
            batch: (todo): write your description
        """
        imgs, pairs_info = [], []
        for unit in batch:
            pairs_info.append([unit['name1'], unit['name2'], unit['label']])
            imgs.append(torch.cat((unit['face1'], unit['face2']), dim=0))
        return (torch.stack(imgs, dim=0), np.array(pairs_info))
    
    
    def _data_loader(self):
        """
        Batch function.

        Args:
            self: (todo): write your description
        """

        self.data['train'] = DataLoader(
                                 dlib.DataBase(self.args),
                                 batch_size=self.args.batch_size, \
                                 shuffle=True,
                                 num_workers=self.args.workers)
        
        self.data['lfw'] = DataLoader(
                               dlib.VerifyBase(self.args, benchmark = 'lfw'),
                               batch_size=self.args.batch_size // 2, \
                               num_workers=self.args.workers,
                               drop_last=False,
                               collate_fn=self.collate_fn_1v1)
        print('Data loading was finished ...')


    def _train_one_epoch(self, epoch = 0):
        """
        Train the model

        Args:
            self: (todo): write your description
            epoch: (int): write your description
        """

        self.model['backbone'].train()
        self.model['fc_layer'].train()

        loss_recorder, batch_acc = [], []
        for idx, (imgs, gtys, _) in enumerate(self.data['train']):

            imgs.requires_grad = False
            gty.requires_grad = False

            if self.use_gpu:
                imgs = imgs.cuda()
                gtys = gtys.cuda()

            outfeat, _, _ = self.model['backbone'](imgs)          # CORE
            mu, logvar = self.model['reg_head'](outfeat)   # CORE
            loss = self.model['criterion'](mu, logvar, gty) # CORE
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            loss_recorder.append(loss.item())
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train']), np.mean(loss_recorder)))
        train_loss = np.mean(loss_recorder)
        print('train_loss : %.4f' % train_loss)
        return train_loss
    
    
    def _save_weights(self, testinfo = {}):
        ''' save the weights during the process of training '''
        
        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)
            
        freq_flag = self.result['epoch'] % self.args.save_freq == 0
        sota_flag = self.result['sota_acc'] < testinfo['test_acc']
        save_name = '%s/epoch_%02d_lfw_opt_thresh_%.4f_racc_%.4f.pth' % \
                         (self.args.save_to, self.result['epoch'], testinfo['opt_thresh'], testinfo['test_acc'])
        if sota_flag:
            save_name = '%s/sota.pth' % self.args.save_to
            self.result['opt_thresh'] = testinfo['opt_thresh']
            self.result['sota_acc']   = testinfo['test_acc']
            print('%s Yahoo, SOTA model was updated %s' % ('*'*16, '*'*16))
        
        if sota_flag or freq_flag:
            torch.save({
                'epoch'   : self.result['epoch'], 
                'backbone': self.model['backbone'].state_dict(),
                'fc_layer': self.model['fc_layer'].state_dict(),
                'thresh'  : testinfo['opt_thresh'],
                'sota_acc': testinfo['test_acc']}, save_name)
            
        if sota_flag and freq_flag:
            normal_name = '%s/epoch_%02d-lfw_opt_thresh_%.4f-racc_%.4f.pth' % \
                              (self.args.save_to, self.result['epoch'], testinfo['opt_thresh'], testinfo['test_acc'])
            shutil.copy(save_name, normal_name)
            
            
    def _dul_training(self):
        """
        Perform training.

        Args:
            self: (todo): write your description
        """

        
        self.result['opt_thresh'] = -1.0
        self.result['sota_acc']   = 0
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()
            self.result['epoch'] = epoch
            train_loss = self._train_one_epoch(epoch)
            self.model['scheduler'].step()
            eval_info = self._evaluate_one_epoch(loader='lfw')
            end_time = time.time()
            print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))
            self._save_weights(eval_info)
            
            if self.args.is_debug:
                break


    def train_runner(self):
        """
        Train the runner.

        Args:
            self: (todo): write your description
        """

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._dul_training()


if __name__ == "__main__":

    dul_reg = DulRegTrainer(reg_args())
    dul_reg.train_runner()
