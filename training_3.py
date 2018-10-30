# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:04:01 2018

@author: Rafael Rocha
"""

import sys
import time
import os
import numpy as np
import keras
import matplotlib.pyplot as plt

#import my_utils as ut

from sklearn.metrics import classification_report, confusion_matrix

from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier

data_set_name = 'train_test_splits_2.npz'
#data_set_name = 'dataset_pad_128x256_aug.npz'

data = np.load(data_set_name)

x = data['x']
y = data['y']
x_test = data['x_test']
y_test = data['y_test']
samples = data['samples']

x_train = x[samples[9]]
y_train = y[samples[9]]

x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
x_test = x_test.reshape(x_test.shape[0], 128, 256, 1)


def build_model_sequential():
#    K.clear_session()
    model = keras.Sequential()

    model.add(Conv2D(32, kernel_size=(11, 11), strides=5, activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=.3),
                  metrics=['accuracy'])
    return model


def build_model():

#    K.clear_session()

    input_shape = (np.size(x_train, 1), np.size(x_train, 2), 1)

    inputs = Input(input_shape)

    # My net
    conv0 = Conv2D(32, kernel_size=(11, 11), strides=5, activation='relu',
                   input_shape=input_shape)(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv1)
#    pool0 = Dropout(0.25)(pool0)
    flatt0 = Flatten()(pool0)
    dense0 = Dense(128, activation='relu')(flatt0)
#    dense0 = Dropout(0.25)(dense0)
    outputs = Dense(3, activation='softmax')(dense0)  # x

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=.3),
                  metrics=['accuracy'])

    return model


# K.clear_session()

input_shape = (np.size(x_train, 1), np.size(x_train, 2), 1)

model_1 = KerasClassifier(build_fn=build_model_sequential,
                          epochs=20,
                          verbose=0)

model_2 = KerasClassifier(build_fn=build_model_sequential,
                          epochs=50,
                          verbose=0)

eclf = VotingClassifier(estimators=[('0', model_1), ('1', model_2)],
                                    voting='hard')


eclf.fit(x_train, y_train)
print(eclf.score(x_test, y_test))

#for clf, label in zip([model_1, model_2, eclf], ['Model 1', 'Model 2', 'Ensemble']):
#    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#
#     clf.fit(x_train, y_train)
#     print(clf.score(x_test, y_test))
