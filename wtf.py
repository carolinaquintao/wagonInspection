# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:27:42 2018

@author: Rafael Rocha
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold

data = np.load('dataset_pad_original.npz')
x = data['x']
y = data['y']

skf = StratifiedKFold(n_splits=5)

for train_index, test_index in skf.split(x, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_test, y_test = x[test_index], y[test_index]
    x, y = x[train_index], y[train_index]
    break

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
for i in range(10):
    samples.append(sample(y))