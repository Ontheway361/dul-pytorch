#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
from torch.utils import data
import torchvision.transforms as T

from IPython import embed

class LFW(object):

    def __init__(self, args, mode = 'test'):

        self.args  = args
        self.mode  = mode
        self.trans = T.Compose([T.ToTensor(), \
                                T.Normalize(mean=[0.5, 0.5, 0.5], \
                                             std=[0.5, 0.5, 0.5])])
        with open(args.pairs_file, 'r') as f:
            self.pairs = np.array(f.readlines())
            shuffleidx = np.random.permutation(len(self.pairs))
            self.pairs = self.pairs[shuffleidx]
        f.close()
        if args.is_debug:
            self.pairs = self.pairs[:1024]
            print('debug version for lfw ...')
        self.num_pairs = len(self.pairs)
        
    
    def _load_imginfo(self, img_name):
        
        img_path = os.path.join(self.args.lfw_dir, img_name)
        img = None
        try:
            img = cv2.resize(cv2.imread(img_path), self.args.in_size)  #  TODO
        except Exception as e:
            img = None
        return img
    
    
    def _get_pair(self, index):

        pair_info = self.pairs[index].strip().split('\t')
        info_dict = {}
        try:
            if 3 == len(pair_info):
                info_dict['label'] = 1
                info_dict['name1'] = pair_info[0] + '/' + pair_info[0] + '_' + '{:04}.jpg'.format(int(pair_info[1]))
                info_dict['name2'] = pair_info[0] + '/' + pair_info[0] + '_' + '{:04}.jpg'.format(int(pair_info[2]))
            elif 4 == len(pair_info):
                info_dict['label'] = 0
                info_dict['name1'] = pair_info[0] + '/' + pair_info[0] + '_' + '{:04}.jpg'.format(int(pair_info[1]))
                info_dict['name2'] = pair_info[2] + '/' + pair_info[2] + '_' + '{:04}.jpg'.format(int(pair_info[3]))

            if info_dict['label'] is not None:
                face1 = self._load_imginfo(info_dict['name1'])
                face2 = self._load_imginfo(info_dict['name2'])
                info_dict['face1'] = self.trans(face1).unsqueeze(0)
                info_dict['face2'] = self.trans(face2).unsqueeze(0)
        except Exception as e:
            for key in ['name1', 'name2', 'label', 'face1', 'face2']:
                info_dict[key] = None
        return info_dict
