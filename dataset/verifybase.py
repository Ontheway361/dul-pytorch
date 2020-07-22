#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from torch.utils import data
import dataset.auglib as auglib

from IPython import embed

class VerifyBase(data.Dataset):

    def __init__(self, args, benchmark = 'lfw'):

        super(VerifyBase).__init__()
        self.args  = args
        self.bmark = benchmark
        self.trans = auglib.aug_naive()
        self.data_dir = ''
        self._load_pairfile()


    def _load_pairfile(self):

        self.pairs = None
        if self.bmark == 'lfw':
            self.data_dir = os.path.join(self.args.lfw_dir, 'align_112_112')
            df_pairs = pd.read_csv(os.path.join(self.args.lfw_dir, 'anno_file/lfw_pairs.csv'))
        else:
            raise TypeError('Only LFW was supported ...')

        self.pairs = df_pairs.to_numpy().tolist()
        if self.args.is_debug:
            self.pairs = self.pairs[:1024]
        self.num_pairs = len(self.pairs)
        print('There are %3d pairs in benchmark-%s' % (self.num_pairs, self.bmark))


    def _load_imginfo(self, img_name):

        img_path = os.path.join(self.data_dir, img_name)
        img = None
        try:
            # img = cv2.resize(cv2.imread(img_path), self.args.in_size[::-1])  #  TODO
            img = cv2.imread(img_path)
        except Exception as e:
            img = None
        return img


    def __getitem__(self, index):

        pair_info = self.pairs[index]
        info_dict = {}
        try:
            info_dict['name1'] = pair_info[0]
            info_dict['name2'] = pair_info[1]
            info_dict['label'] = pair_info[-1]
            face1 = self._load_imginfo(info_dict['name1'])
            face2 = self._load_imginfo(info_dict['name2'])
            info_dict['face1'] = self.trans(image=face1)['image'].unsqueeze(0)
            info_dict['face2'] = self.trans(image=face2)['image'].unsqueeze(0)
        except:
            for key in ['name1', 'name2', 'label', 'face1', 'face2']:
                info_dict[key] = None
        return info_dict

    def __len__(self):
        return self.num_pairs