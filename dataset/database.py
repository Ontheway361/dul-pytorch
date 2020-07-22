#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
from torch.utils import data
import dataset.auglib as auglib

from IPython import embed


class DataBase(data.Dataset):

    def __init__(self, args):

        self.args       = args
        self.transforms = auglib.aug_naive()
        
        with open(args.train_file, 'r') as f:
            self.lines  = f.readlines()
        f.close()
        if args.is_debug:
            self.lines = self.lines[:1024]  # just for debug
            print('debug version for ms1m-mini ...')
    
    
    def _load_imginfo(self, img_name):
        
        img_path = os.path.join(self.args.ms1m_dir, img_name)
        img = None
        try:
            img = cv2.imread(img_path)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
        except Exception as e:
            img = None
        return img
    

    def __getitem__(self, index):

        info = self.lines[index].strip().split(' ')
        img  = self._load_imginfo(info[0])
        while img is None:
            idx  = np.random.randint(0, len(self.lines) - 1)
            info = self.lines[idx].strip().split(' ')
            img  = self._load_imginfo(info[0])
        img = self.transforms(image=img)['image']
        return (img, int(info[1]), info[0])

    
    def __len__(self):
        return len(self.lines)
