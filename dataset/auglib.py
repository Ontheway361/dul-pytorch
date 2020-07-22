#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import albumentations as alt
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2 as ToTensor


def aug_train():
    ''' augmentation for training a FR-system '''
    aug = alt.Compose([
              alt.Resize(height=112, width=112),
              alt.HorizontalFlip(p=0.5),
              alt.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
              alt.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
              alt.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
              alt.ToGray(p=0.01),
              alt.MotionBlur(blur_limit=7, p=0.2),   # default=11
              alt.GaussianBlur(blur_limit=7, p=0.2), # default=11
              alt.GaussNoise(var_limit=(5.0, 20.0), mean=0, p=0.1), # defalt var_limit=(10.0, 30.0)
              alt.ISONoise(p=0.2),
              alt.Normalize(),
              ToTensor()])
    return aug

def aug_infer():
    ''' augmentation for inference of a FR-system '''
    return alt.Compose([alt.Normalize(), ToTensor()])


def aug_naive():
    
    return alt.Compose([
               alt.Resize(height=112, width=112),
               alt.Normalize(),
               ToTensor()])

def aug_old():
    return transforms.Compose([
               transforms.Resize(size=(112, 112)), \
               transforms.ToTensor(), \
               transforms.Normalize(mean=[0.5], std=[0.5])])