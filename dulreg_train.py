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
from config import training_args

torch.backends.cudnn.bencmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" # TODO

from IPython import embed


class DULTrainer(mlib.Faster1v1):

    def __init__(self, args):

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

        self.model['backbone']  = mlib.dulres_zoo(self.args.backbone, drop_ratio=self.args.drop_ratio, use_se=self.args.use_se)  # ResBlock
        self.model['fc_layer']  = mlib.FullyConnectedLayer(self.args)
        self.model['criterion'] = mlib.DULLoss(self.args)
        self.model['optimizer'] = torch.optim.SGD(
                                      [{'params': self.model['backbone'].parameters()},
                                       {'params': self.model['fc_layer'].parameters()}],
                                      lr=self.args.base_lr,
                                      weight_decay=self.args.weight_decay,
                                      momentum=0.9,
                                      nesterov=True)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], milestones=self.args.lr_adjust, gamma=self.args.gamma)
        if self.use_gpu:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['fc_layer']  = self.model['fc_layer'].cuda()
            self.model['criterion'] = self.model['criterion'].cuda()

        if self.use_gpu and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['fc_layer'] = torch.nn.DataParallel(self.model['fc_layer'], device_ids=self.args.gpu_ids)
            print('Parallel mode was going ...')
        elif self.use_gpu:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['fc_layer'].load_state_dict(checkpoint['fc_layer'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')

    
    @staticmethod
    def collate_fn_1v1(batch):
        imgs, pairs_info = [], []
        for unit in batch:
            pairs_info.append([unit['name1'], unit['name2'], unit['label']])
            imgs.append(torch.cat((unit['face1'], unit['face2']), dim=0))
        return (torch.stack(imgs, dim=0), np.array(pairs_info))
    
    
    def _data_loader(self):

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

        self.model['backbone'].train()
        self.model['fc_layer'].train()

        loss_recorder, batch_acc = [], []
        for idx, (img, gty, _) in enumerate(self.data['train']):

            img.requires_grad = False
            gty.requires_grad = False

            if self.use_gpu:
                img = img.cuda()
                gty = gty.cuda()

            mu, logvar, embedding = self.model['backbone'](img)
            output  = self.model['fc_layer'](embedding, gty)
            loss    = self.model['criterion'](output, gty, mu, logvar)
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            predy   = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
            it_acc  = np.mean((predy == gty.data.cpu().numpy()).astype(int))
            batch_acc.append(it_acc)
            loss_recorder.append(loss.item())
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f, batch_ave_acc : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train']), np.mean(loss_recorder), np.mean(batch_acc)))
        train_loss = np.mean(loss_recorder)
        print('train_loss : %.4f' % train_loss)
        return train_loss
    
    
    def _save_weights(self, testinfo = {}):
        ''' save the weights during the process of training '''
        
        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)
            
        freq_flag = self.result['epoch'] % self.args.save_freq == 0
        sota_flag = self.result['sota_acc'] < testinfo['test_acc']
        save_name = '%s/epoch_%02d-lfw_opt_thresh_%.4f-racc_%.4f.pth' % \
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

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._dul_training()


if __name__ == "__main__":

    dul = DULTrainer(training_args())
    dul.train_runner()
