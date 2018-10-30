# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:04:01 2018

@author: Rafael Rocha
"""

import numpy as np
import keras

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

data_set_name = 'train_test_splits_3.npz'

data = np.load(data_set_name)
x = data['x']
y = data['y']
train_index = data['train_index']
test_index = data['test_index']
# samples = data['samples']

x_train, y_train = x[train_index], y[train_index]
x_test, y_test = x[test_index], y[test_index]
# x_train_sample, y_train_sample = x_train[samples[0]], y_train[samples[0]]
# x_test = data['x_test']
# y_test = data['y_test']

x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
x_test = x_test.reshape(x_test.shape[0], 128, 256, 1)

input_shape = (np.size(x_train, 1), np.size(x_train, 2), 1)


def build_model_sequential_1():
    model = keras.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu',
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


def build_model_sequential_2():
    model = keras.Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=1, activation='relu',
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


def build_model_sequential_3():
    model = keras.Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=5, activation='relu',
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


def build_model_sequential_4():
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


def build_model_sequential_5():
    # lenet
    model = keras.Sequential()
    model.add(Conv2D(20, kernel_size=(5, 5), strides=1,
                     padding='same', activation='tanh',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=(5, 5), strides=(
        5, 5), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.3),
                  metrics=['accuracy'])
    return model


def build_model_sequential_6():
    # lenet_5
    model = keras.Sequential()
    model.add(Conv2D(20, kernel_size=(5, 5), strides=5,
                     padding='same', activation='tanh',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=(5, 5), strides=(
        5, 5), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.3),
                  metrics=['accuracy'])

    return model


# results = []
model_1 = KerasClassifier(build_fn=build_model_sequential_6,
                          epochs=20,
                          verbose=1)
model_1.fit(x_train, y_train)
y_pred = model_1.predict(x_test)
print(classification_report(y_test, y_pred, digits=4))
print(confusion_matrix(y_test, y_pred))
# results.append(y_pred)
print(y_pred)

#np.savetxt('Lenet_results_100_7.txt', [y_pred], fmt='%i', delimiter=',')
# model_1.fit()
