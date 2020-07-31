#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

root_dir = '/home/jovyan/jupyter/benchmark_images/faceu'
lfw_dir  = osp.join(root_dir, 'face_verfication/lfw')
ms1m_dir = osp.join(root_dir, 'face_recognition/ms1m_mini')
cp_dir   = '/home/jovyan/jupyter/checkpoints_zoo/face-recognition/densityEstimate/experiments_dul'

def reg_args():

    parser = argparse.ArgumentParser(description='PyTorch for DUL-regression')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--workers', type=int,  default=0)  # TODO

    # -- model
    parser.add_argument('--backbone',   type=str,    default='dulres18')   # paper-setting
    parser.add_argument('--use_se',     type=bool,   default=True)         # IRESSE
    parser.add_argument('--used_as',    type=bool,   default='backbone')
    parser.add_argument('--in_feats',   type=int,    default=512)
    parser.add_argument('--drop_ratio', type=float,  default=0.4)          # TODO
    parser.add_argument('--classnum',   type=int,    default=85164)        # MS1M-mimi

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        #
    parser.add_argument('--end_epoch',   type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=256)      # TODO | 300
    parser.add_argument('--base_lr',     type=float, default=0.01)     # paper-setting
    parser.add_argument('--lr_adjust',   type=list,  default=[4, 6])
    parser.add_argument('--gamma',       type=float, default=0.1)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=5e-4)     # FIXED
    parser.add_argument('--resume',      type=str,   default=osp.join(cp_dir, 'dul_paper_setting/sota.pth'))       # checkpoint

    # -- dataset
    parser.add_argument('--ms1m_dir',   type=str, default=ms1m_dir)   # 
    parser.add_argument('--lfw_dir',    type=str, default=lfw_dir)    # TODO
    parser.add_argument('--train_file', type=str, default=osp.join(ms1m_dir, 'anno_file/ms1m_mini_3.3_million.txt')) # 3314259-lines

    # -- verification
    parser.add_argument('--n_folds',   type=int,   default=10)
    parser.add_argument('--thresh_iv', type=float, default=0.005)

    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=True)   # TODO
    parser.add_argument('--save_to',   type=str,   default=osp.join(cp_dir, 'dul_paper_setting'))
    parser.add_argument('--print_freq',type=int,   default=1000)  # {'bz-512': 6474}
    parser.add_argument('--save_freq', type=int,   default=3)  # TODO

    args = parser.parse_args()

    return args
