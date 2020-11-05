#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

root_dir = '/data/relu/benchmark_images/faceu'
test_dir = osp.join(root_dir, 'faces_verification')
data_dir = osp.join(root_dir, 'ms1m_arcface')
cp_dir   = '/data/relu/checkpoint/face-recognition/densityEstimate/experiments_dul'

def cls_args():
    """
    Parse command line arguments.

    Args:
    """

    parser = argparse.ArgumentParser(description='PyTorch for DUL-classification')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--workers', type=int,  default=16)  # TODO

    # -- model
    parser.add_argument('--backbone',   type=str,   default='dulres18')   # paper-setting
    parser.add_argument('--use_se',     type=bool,  default=True)         # IRESSE
    parser.add_argument('--drop_ratio', type=float, default=0.4)          # TODO
    parser.add_argument('--used_as',    type=str,   default='dul_cls', choices=['baseline', 'dul_cls', 'backbone'])   
    parser.add_argument('--in_feats',   type=int,   default=512)
    parser.add_argument('--classnum',   type=int,   default=85164)        # MS1M-arcface

    # -- fc-layer
    parser.add_argument('--t',          type=float,  default=0.2)          # MV
    parser.add_argument('--margin',     type=float,  default=0.5)          # [0.5-arcface, 0.35-cosface]
    parser.add_argument('--easy_margin',type=bool,   default=True)
    parser.add_argument('--scale',      type=float,  default=64)           # FIXED
    parser.add_argument('--kl_lambda',  type=float,  default=0)         # default = 0.01
    parser.add_argument('--fc_mode',    type=str,    default='arcface',  choices=['softmax', 'sphere', 'cosface', 'arcface', 'mvcos', 'mvarc'])
    parser.add_argument('--hard_mode',  type=str,    default='adaptive', choices=['fixed', 'adaptive']) # MV
    parser.add_argument('--loss_mode',  type=str,    default='ce',       choices=['ce', 'focal_loss', 'hardmining'])
    parser.add_argument('--hard_ratio', type=float,  default=0.9)          # hardmining
    parser.add_argument('--loss_power', type=int,    default=2)            # focal_loss

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        #
    parser.add_argument('--end_epoch',   type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=512)      # TODO | 300
    parser.add_argument('--base_lr',     type=float, default=0.1)      # default = 0.1
    parser.add_argument('--lr_adjust',   type=list,  default=[12, 20, 30, 42])
    parser.add_argument('--gamma',       type=float, default=0.3)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=5e-4)     # FIXED
    parser.add_argument('--resume',      type=str,   default='')       # checkpoint

    # -- dataset
    parser.add_argument('--data_dir',   type=str,  default=data_dir)   # 
    parser.add_argument('--test_dir',   type=str,  default=test_dir)   # TODO
    parser.add_argument('--bmark_list', type=list, default=['lfw'])    # ['lfw', 'agedb30', 'cfp_ff', 'cfp_fp']
    parser.add_argument('--train_file', type=str,  default=osp.join(data_dir, 'anno_file/ms1m_images.txt')) # 3314259-lines

    # -- verification
    parser.add_argument('--n_folds',   type=int,   default=10)
    parser.add_argument('--thresh_iv', type=float, default=0.005)

    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=False)   # TODO
    parser.add_argument('--save_to',   type=str,   default=osp.join(cp_dir, 'res18IRSE_arcface_ms1m_dulcls'))
    parser.add_argument('--print_freq',type=int,   default=2400)  # (3804846, 512, 7432)
    parser.add_argument('--save_freq', type=int,   default=3)  # TODO

    args = parser.parse_args()

    return args
