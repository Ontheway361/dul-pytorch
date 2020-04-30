#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

# root_dir = '/Users/relu/data/benchmark_images/faceu/CASIA-WebFace'   # mini-mac
root_dir = '/home/jovyan/lujie/gpu3/benchmark_images/faceu/lfw'

def verify_args():

    parser = argparse.ArgumentParser(description='PyTorch metricface')

    # -- env
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=list, default=[0,1])
    parser.add_argument('--workers',  type=int,  default=0)

    # -- model
    parser.add_argument('--classnum', type=int,  default=10574)  # CASIA
    parser.add_argument('--resume',   type=str,  default='../checkpoints/sphere20a/sota_0.9867.pth.tar')

    # -- dataset
    parser.add_argument('--in_size',    type=tuple,default=(112, 96))
    parser.add_argument('--lfw_dir',    type=str,  default=osp.join(root_dir, 'align_112_96'))  # un-aligned
    parser.add_argument('--pairs_file', type=str,  default=osp.join(root_dir, 'anno_file/pairs.txt'))
    parser.add_argument('--lmk_files',  type=str, default=osp.join(root_dir, 'anno_file/lfw_landmark.txt'))

    # -- verbose
    parser.add_argument('--is_debug',    type=bool,default=True)
    parser.add_argument('--n_folds',     type=int, default=2)
    parser.add_argument('--thresh_iv', type=float, default=0.005)

    args = parser.parse_args()

    return args
