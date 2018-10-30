#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:15:26 2017

@author: Rafael Rocha
"""

import os
import numpy as np
# import my_utils as ut
from glob import glob
from skimage.transform import resize
from skimage.io import imread
from skimage.exposure import equalize_hist
from skimage import img_as_ubyte, img_as_float
# from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


img_dir = '/Users/pma009/Documents/Banco_de_imagens/MeliponasImageDataStore/'
dir_prefix = '*/'
file_prefix = '*.jpeg'
# img_rows, img_cols = 128, 256

# x, y = ut.extract_2(img_dir, dir_prefix, file_prefix)

dir_list = glob(os.path.join(img_dir, dir_prefix))
img_list = []
labels_list = []
lin = 256
col = 512
# print(range(np.size(dir_list)))

for i in range(np.size(dir_list)):
    for filename in glob(os.path.join(dir_list[i], file_prefix)):
        im = imread(filename)
        im = resize(im, [lin, col, 3])
        im = equalize_hist(im)
        im = img_as_ubyte(im)
        im = img_as_float(im)
        img_list.append(im)
        train_len = len(dir_list[i])
        labels_list.append(i)#dir_list[i][train_len - 2])#y_true[i])#
        # print(i)

#    var_perm = np.random.permutation(np.size(labels_list))
X = np.array(img_list, dtype=np.float64)
y = np.array(labels_list, dtype=np.uint8)

# x, x_test, y, y_test = train_test_split(x, y, test_size=.2, shuffle=False)

#
# skf = StratifiedKFold(n_splits=5, shuffle=False)
#
# for train_index, test_index in skf.split(x, y):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_test, y_test = x[test_index], y[test_index]
#     x, y = x, y#x[train_index], y[train_index]
#     break
skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)
shufflesplit = StratifiedShuffleSplit(n_splits=2, random_state=42)

skf.get_n_splits(X, y)
print(skf)  # doctest: +NORMALIZE_WHITESPACE

shufflesplit.get_n_splits(X, y)
print(shufflesplit)
# StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    # for train_index, test_index in skf.split(X, y):
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    # print(y_test)

for train_index, test_index in shufflesplit.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
print(y_test)

def sample(y):
    d_c = np.bincount(y)
    s = int(d_c.min() * 0.8)
    a1 = np.random.choice(np.arange(d_c[0]), size=s, replace=False)
    a2 = np.random.choice(
        np.arange(d_c[0], d_c[0] + d_c[1]), size=s, replace=False)
    a3 = np.random.choice(
        np.arange(d_c[0] + d_c[1], sum(d_c)), size=s, replace=False)
    a = np.concatenate([a1, a2, a3])
    print(d_c)
    return a


samples = []
for i in range(5):
    samples.append(sample(y))

np.savez('train_test_splits_Meliponas8especies256x512_2', samples=samples,
        x=X, y=y, x_test=x_test, y_test=y_test, train_index=train_index, test_index=test_index)
