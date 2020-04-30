#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import random
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F

import model as model_lib

torch.backends.cudnn.bencmark = True

from IPython import embed


class VerifyFace(object):

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.device = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python: {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch: {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- device: {}".format(self.device))
        print('-' * 52)


    def _model_loader(self):

        self.model['backbone'] = model_lib.resnet_zoo(self.args.backbone)
        if self.device:
            self.model['backbone'] = self.model['backbone'].cuda()

        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            torch.backends.cudnn.benchmark = True
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage.cuda(0))
            if (self.device and len(self.args.gpu_ids) > 1) and ('cpu' in self.args.resume):
                self.model['backbone'].module.load_state_dict(checkpoint)
            else:
                try:
                    self.args.start_epoch = checkpoint['epoch']
                except:
                    self.args.start_epoch = 0
                    print('there is no start epoch info in checkpoint ...')
                self.model['backbone'].load_state_dict(checkpoint['backbone'])
            print('Resuming the train process at %2d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _eval_lfw(self):
        
        self.model['backbone'].eval()  # CORE
        with torch.no_grad():
            simi_list = []
            for index in range(1, self.data['lfw'].num_pairs):
                
                try:
                    pair_dict = self.data['lfw']._get_pair(index)
                    if pair_dict['label'] is not None:
                        if self.device:
                            pair_dict['face1'] = pair_dict['face1'].cuda()
                            pair_dict['face2'] = pair_dict['face2'].cuda()
                        feat1, _ = self.model['backbone'](pair_dict['face1'])   # TODO
                        feat2, _ = self.model['backbone'](pair_dict['face2'])   # TODO
                        cosvalue = feat1[0].dot(feat2[0]) / (feat1[0].norm() * feat2[0].norm() + 1e-5)
                        simi_list.append([pair_dict['name1'], pair_dict['name2'], pair_dict['label'], cosvalue.item()])
                except:
                    pass
                # if (index + 1) % 500 == 0:
                    # print('alreay processed %3d, total %3d' % (index+1, self.data['lfw'].num_pairs))
            self.data['similist'] = np.array(simi_list)
            # col_name = ['name1', 'name2', 'gt_y', 'pred_y']
            # df_simi = pd.DataFrame(self.data['similist'], columns=col_name)
            # df_simi.to_csv('check_2.csv')
            # print('lfw-pair faces was evaluated, there are %3d paris' % len(simi_list))
        
        
    def _eval_aku8k(self):
        ''' Design for the raw-images, which equiped with a pairfile.csv '''

        self.model['backbone'].eval()
        with torch.no_grad():
            
            simi_list = []
            for index in range(1, self.data['aku8k'].num_pairs):
                
                try:
                    pair_dict = self.data['aku8k']._get_pair(index)
                    if pair_dict['label'] is not None:
                        if self.device:
                            pair_dict['face1'] = pair_dict['face1'].cuda()
                            pair_dict['face2'] = pair_dict['face2'].cuda()
                        feat1 = self.model['backbone'](pair_dict['face1'])
                        feat2 = self.model['backbone'](pair_dict['face2'])
                        cosvalue = feat1[0].dot(feat2[0]) / (feat1[0].norm() * feat2[0].norm() + 1e-5)
                        simi_list.append([pair_dict['name1'], pair_dict['name2'], pair_dict['label'], cosvalue.item()])
                except:
                    pass
                # if (index + 1) % 500 == 0:
                    # print('alreay processed %3d, total %3d' % (index+1, self.data['lfw'].num_pairs))
            self.data['similist'] = np.array(simi_list)
        


    def _k_folds(self):

        num_lines = len(self.data['similist'])
        folds, base = [], list(range(num_lines))
        for k in range(self.args.n_folds):

            start = int(k * num_lines / self.args.n_folds)
            end   = int((k + 1) * num_lines / self.args.n_folds)
            test  = base[start : end]
            train = list(set(base) - set(test))
            folds.append([train, test])
        self.data['folds'] = folds


    def _cal_acc(self, index, thresh):

        gt_y, pred_y = [], []
        for row in self.data['similist'][index]:
            
            same = 1 if float(row[-1]) > thresh else 0
            pred_y.append(same)
            gt_y.append(int(row[-2]))
        gt_y = np.array(gt_y)
        pred_y = np.array(pred_y)
        accuracy = 1.0 * np.count_nonzero(gt_y==pred_y) / len(gt_y)
        return accuracy


    def _find_best_thresh(self, train, test):

        best_thresh, best_acc = 0, 0
        for thresh in np.arange(-1, 1, self.args.thresh_iv):

            acc = self._cal_acc(train, thresh)
            if best_acc < acc:
                best_acc = acc
                best_thresh = thresh
        test_acc = self._cal_acc(test, best_thresh)
        return (best_thresh, test_acc)


    def _eval_runner(self):

        opt_thresh_list, test_acc_list = [], []
        for k in range(self.args.n_folds):

            train, test = self.data['folds'][k]
            best_thresh, test_acc = self._find_best_thresh(train, test)
            print('fold : %2d, thresh : %.3f, test_acc : %.4f' % (k, best_thresh, test_acc))
            opt_thresh_list.append(best_thresh)
            test_acc_list.append(test_acc)

        opt_thresh = np.mean(opt_thresh_list)
        test_acc   = np.mean(test_acc_list)
        print('verification was finished, best_thresh : %.4f, test_acc : %.4f' % (opt_thresh, test_acc))
        return opt_thresh, test_acc


    def verify_runner(self):

        self._report_settings()

        self._model_loader()

        self._eval_lfw()

        self._k_folds()

        self._eval_runner()

