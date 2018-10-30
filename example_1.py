# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:20:08 2018

@author: Rafael Rocha
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

data1 = np.load('dataset_pad_original.npz')
data2 = np.load('splits.npz')

x = data1['x']
y = data1['y']

splits = data2['splits']

#print(np.size(splits[0]))

#x_train = []
#y_train = []

for i in range(np.size(splits)):
    split = splits[i]
    print(i)

plt.plot(splits[0])
plt.plot(splits[1])
plt.plot(splits[2])
plt.plot(splits[3])
plt.plot(splits[4])

a = np.concatenate((splits[1], splits[2], splits[3], splits[4]), axis=0)
x_train_1 = x[a]
y_train_1 =  y[a]
x_test_1 = x[splits[0]]
y_test_1 = y[splits[0]]

a = np.concatenate((splits[0], splits[2], splits[3], splits[4]), axis=0)
x_train_2 = x[a]
y_train_2 =  y[a]
x_test_2 = x[splits[1]]
y_test_2 = y[splits[1]]

a = np.concatenate((splits[0], splits[1], splits[3], splits[4]), axis=0)
x_train_3 = x[a]
y_train_3 =  y[a]
x_test_3 = x[splits[2]]
y_test_3 = y[splits[2]]

a = np.concatenate((splits[0], splits[1], splits[2], splits[4]), axis=0)
x_train_4 = x[a]
y_train_4 =  y[a]
x_test_4 = x[splits[3]]
y_test_4 = y[splits[3]]

a = np.concatenate((splits[0], splits[1], splits[2], splits[3]), axis=0)
x_train_5 = x[a]
y_train_5 =  y[a]
x_test_5 = x[splits[4]]
y_test_5 = y[splits[4]]

np.savez('train_test_splits', x_train_1=x_train_1, y_train_1=y_train_1, x_test_1=x_test_1, y_test_1=y_test_1,
         x_train_2=x_train_2, y_train_2=y_train_2, x_test_2=x_test_2, y_test_2=y_test_2,
         x_train_3=x_train_3, y_train_3=y_train_3, x_test_3=x_test_3, y_test_3=y_test_3,
         x_train_4=x_train_4, y_train_4=y_train_4, x_test_4=x_test_4, y_test_4=y_test_4,
         x_train_5=x_train_5, y_train_5=y_train_5, x_test_5=x_test_5, y_test_5=y_test_5 )