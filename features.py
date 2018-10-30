# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:14:22 2018

@author: Rafael Rocha
"""

import os
import numpy as np
import pywt
import cv2
from skimage.io import imread, imshow
from skimage.exposure import equalize_hist
from skimage import img_as_ubyte, img_as_float
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog

from glob import glob

img_dir = 'original/'
dir_prefix = '*/'
file_prefix = '*.png'

dir_list =  glob(os.path.join(img_dir, dir_prefix))
img_list = []
labels_list = []

var_list = []

for i in range(np.size(dir_list)):
    for filename in glob(os.path.join(dir_list[i], file_prefix)):
        img = imread(filename)
        img = np.array(img, dtype=np.float64)
        
        img_list.append(img)
        train_len = len(dir_list[i])
        labels_list.append(dir_list[i][train_len-2])
        
#    var_perm = np.random.permutation(np.size(labels_list))
x = np.asarray(img_list)
#x = np.array(img_list, dtype=np.float64)
y = np.array(labels_list, dtype=np.uint8)