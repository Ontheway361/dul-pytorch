#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
import torchvision
from torch.utils import data

from IPython import embed


class DataBase(data.Dataset):

    def __init__(self, args, mode = 'train'):

        self.args       = args
        self.mode       = mode
        self.transforms = torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], \
                                                                std=[0.5, 0.5, 0.5])])
        with open(args.train_file, 'r') as f:
            self.lines  = f.readlines()
        f.close()
        if args.is_debug:
            self.lines = self.lines[:1024]  # just for debug
            print('debug version for casia ...')
    
    
    def _load_imginfo(self, img_name):
        
        img_path = os.path.join(self.args.casia_dir, img_name)
        img = None
        try:
            img = cv2.resize(cv2.imread(img_path), self.args.in_size)  #  TODO
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
        img = self.transforms(img)
        return (img, int(info[1]), info[0])

    
    def __len__(self):
        return len(self.lines)
