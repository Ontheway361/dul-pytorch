#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from sklearn import metrics
from IPython import embed

class Faster1v1(object):
    ''' Speed up the module of the face verification during the training '''

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.device = args.use_gpu and torch.cuda.is_available()


    def _model_loader(self):
        pass


    def _data_loader(self):
        pass

    def _quickly_1v1(self, loader = 'lfw'):

        self.model['backbone'].eval()
        with torch.no_grad():

            simi_list   = []
            for idx, (imgs, pair_info) in enumerate(self.data[loader]):

                imgs = imgs.view((-1, 3, imgs.size(-2), imgs.size(-1)))
                if self.device:
                    imgs = imgs.cuda()
                feats, _, _ = self.model['backbone'](imgs)
                norms = torch.unsqueeze(torch.norm(feats, dim=1), dim=1)
                norms = torch.mm(norms, norms.t())
                simis = torch.mm(feats, feats.t()) / (norms + 1e-5)
                pindx = np.array([2 * i for i in range(len(pair_info))])
                simis = simis.data.cpu().numpy()[pindx, pindx + 1][:, np.newaxis]
                simi_list.extend(np.concatenate((pair_info, simis), axis=1).tolist())
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


    def calculate_acc(self, opt_thresh = 0.45):

        gt_y       = np.array(self.data['similist'][:, -2], dtype=int)
        pred_y     = (np.array(self.data['similist'][:, -1], np.float) >= opt_thresh) * 1
        auc        = metrics.roc_auc_score(gt_y, pred_y)
        acc        = metrics.accuracy_score(gt_y, pred_y)
        recall     = metrics.recall_score(gt_y, pred_y)
        f1_score   = metrics.f1_score(gt_y, pred_y)
        precision  = metrics.precision_score(gt_y, pred_y)
        print('auc : %.4f, acc : %.4f, precision : %.4f, recall : %.4f, f1_score : %.4f' % \
                  (auc, acc, precision, recall, f1_score))

        print('%s gt vs. pred %s' % ('-' * 36, '-' * 36))
        print(metrics.classification_report(gt_y, pred_y, digits=4))
        print(metrics.confusion_matrix(gt_y, pred_y))
        print('-' * 85)
        return acc


    def _evaluate_one_epoch(self, loader = 'lfw'):

        print('quicky 1v1 on %s is going ...' % loader)
        self._quickly_1v1(loader)
        self._k_folds()
        opt_thresh_list, test_acc_list = [], []
        for k in range(self.args.n_folds):

            train, test = self.data['folds'][k]
            best_thresh, test_acc = self._find_best_thresh(train, test)
            opt_thresh_list.append(best_thresh)
            test_acc_list.append(test_acc)
            print('fold : %2d, thresh : %.3f, test_acc : %.4f' % (k, best_thresh, test_acc))
        eval_info = {}
        eval_info['opt_thresh'] = np.mean(opt_thresh_list)
        eval_info['test_acc']   = self.calculate_acc(eval_info['opt_thresh'])
        print('verification was finished, best_thresh : %.4f, test_acc : %.4f' % \
              (eval_info['opt_thresh'], eval_info['test_acc']))
        return eval_info
