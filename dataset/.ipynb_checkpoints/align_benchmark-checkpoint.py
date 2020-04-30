#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
from alignface import AlignFace

from IPython import embed

folder = ['casia_webface', 'lfw_data']
root_dir = '/home/jovyan/lujie/gpu3/benchmark_images/faceu/' + folder[0]  # TODO


def align_args():
    
    parser = argparse.ArgumentParser(description='Preprocess for face alignment')
    
    parser.add_argument('--root_dir',  type=str,   default=root_dir)
    parser.add_argument('--img_folder',type=str,   default='images',)      # TODO
    parser.add_argument('--lmk_file',  type=str,   default='anno_file/casia_landmark.txt') # TODO :: lfw_ or casia_
    parser.add_argument('--in_size',   type=tuple, default=(128, 128))  # TODO
    parser.add_argument('--idxoff',    type=int,   default=2)    # TODO :: lfw = 1, casia = 2 
    parser.add_argument('--print_freq',type=int,   default=1000)
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
     
    args = align_args()
    
    file = os.path.join(args.root_dir, args.lmk_file)
    with open(file, 'r') as f:
        files = f.readlines()
    f.close()

    loader = AlignFace(in_size=args.in_size, offset=args.offset)
    data_dir = os.path.join(args.root_dir, args.img_folder)
    data_list = []
    start_time = time.time()
    save_to = os.path.join(args.root_dir, 'align_%d_%d' % \
                           (args.in_size[0], args.in_size[1]))
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    print('save_to : %s' % save_to)
    for idx, file in enumerate(files):
        
        ori_file = file
        try:
            file = file.strip().split('\t')
            img_path = os.path.join(data_dir, file[0])
            img = cv2.imread(img_path)
            src_pts = []
            for i in range(5):
                src_pts.append([int(file[2*i+args.idxoff]), int(file[2*i+args.idxoff+1])])   # TODO
            alimg = loader._alignment(img, src_pts)
            folder, img_name = file[0].split('/')
            who = os.path.join(save_to, folder)
            if not os.path.exists(who):
                os.mkdir(who)
            cv2.imwrite(os.path.join(who, img_name), alimg)
        except Exception as e:
            print(e)
            print(ori_file)
        else:
            data_list.append(ori_file)
        
        if (idx + 1) % args.print_freq == 0:
            end_time = time.time()
            print('already processed %4d, total %5d, cost time : %.2fmins' % \
                  (idx+1, len(files), ((end_time-start_time) / 60)))
        
