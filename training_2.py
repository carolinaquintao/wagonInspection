# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:04:32 2018

@author: Rafael Rocha
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import time
import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.losses import categorical_crossentropy

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras import backend as K

np.random.seed(42)

data_set_name = 'train_test_splits_2.npz'
#data_set_name = 'dataset_pad_128x256_aug.npz'

data = np.load(data_set_name)

x = data['x']
y = data['y']
x_test = data['x_test']
y_test = data['y_test']
samples = data['samples']

x_test = x_test.reshape(x_test.shape[0], 128, 256, 1)
y_test = keras.utils.to_categorical(y_test, 3)

if data_set_name is 'dataset_pad_28x32.npz':
    conv0_ks = (3, 3)
else:
    conv0_ks = (3, 3)

result = []
    
batch_sizes = range(2)

for i in batch_sizes:
    print('\n'+'Batch size: '+str(i)+'\n')
    
    K.clear_session()
    
    x_train = x[samples[i]]
    y_train = y[samples[i]]
    
    x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
    
    y_train = keras.utils.to_categorical(y_train, 3)
    
    input_shape = (np.size(x_train, 1), np.size(x_train, 2), 1)
    
    inputs = Input(input_shape)
    
    # My net
    conv0 = Conv2D(32, kernel_size=(11, 11), strides=5, activation='relu',
               input_shape=input_shape)(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv1)
#    pool0 = Dropout(0.25)(pool0)
    flatt0 = Flatten()(pool0)
    dense0 = Dense(128, activation='relu')(flatt0)
#    dense0 = Dropout(0.25)(dense0)
    outputs = Dense(3, activation='softmax')(dense0) # x
    
#    # Lenet5
#    conv0 = Conv2D(20, kernel_size=(11, 11), strides=(5, 5), padding='same',
#                   activation='tanh', input_shape=input_shape)(inputs)
##    conv0 = BatchNormalization()(conv0) 
#    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
#    conv1 = Conv2D(50, kernel_size=5, strides=5, padding='same',
#                   activation='tanh')(pool0)
##    conv1 = BatchNormalization()(conv1)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#    flatt0 = Flatten()(pool1)
#    dense0 = Dense(500, activation='tanh')(flatt0)
##    dense0 = BatchNormalization()(dense0)
#    outputs = Dense(3, activation='softmax')(dense0)
#    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=.3),
                  metrics=['accuracy'])
    
#    h = model.fit(x_train,
#                  y_train,
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  verbose=1)
    
#    result.append(h.history)
    
    start_time = time.time()
    h = model.fit(x_train,
                  y_train,
                  batch_size=5,
                  epochs=50,
                  verbose=1)
    training_time = time.time() - start_time

    score = model.evaluate(x_train, y_train, verbose=0)
    
#    del x_train

    print("\n--- Training time: %s seconds ---" % training_time)
    print('Traning loss:', score[0])
    print('Training accuracy:', score[1])
    
    result.append(h.history)
    
    start_time = time.time()
    score = model.evaluate(x_test,
                           y_test,
                           verbose=0)
    test_time = time.time() - start_time
    
    print("\n--- Test time: %s seconds ---" % test_time)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    y_pred = model.predict(x_test)
    #y_pred = model.predict_classes(x_test)
    
    model_name = 'model_' + str(i)
    model.save(model_name)
    
#    del x_test
    
    target_names = ['Absent', 'Undamaged', 'Damaged']
    
    print('\n')
    
    cr = classification_report(np.argmax(y_test, axis=1),
                               np.argmax(y_pred, axis=1),
                               target_names=target_names,
                               digits=4)
    print(cr)
    
    print('\nConfusion matrix:\n')
    cm = confusion_matrix(np.argmax(y_test, axis=1), 
                          np.argmax(y_pred, axis=1))
    print(cm)
    
    str_acc = "%.2f" % (100*score[1])


x = np.arange(1, np.size(result[0]['acc'])+1)

plt.figure()
plt.plot(x, result[0]['loss'])
plt.plot(x, result[1]['loss'])
plt.plot(x, result[2]['loss'])
plt.plot(x, result[3]['loss'])
plt.plot(x, result[4]['loss'])
plt.plot(x, result[5]['loss'])
plt.plot(x, result[6]['loss'])
plt.plot(x, result[7]['loss'])
plt.plot(x, result[8]['loss'])
plt.plot(x, result[9]['loss'])


plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(x,
        loc='best')
#plt.savefig('teste1')

plt.figure()
plt.plot(x, result[0]['acc'])
plt.plot(x, result[1]['acc'])
plt.plot(x, result[2]['acc'])
plt.plot(x, result[3]['acc'])
plt.plot(x, result[4]['acc'])
plt.plot(x, result[5]['acc'])
plt.plot(x, result[6]['acc'])
plt.plot(x, result[7]['acc'])
plt.plot(x, result[8]['acc'])
plt.plot(x, result[9]['acc'])

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(x,
            loc='best')
#plt.savefig('teste2')

#np.savez(name_file, result=result)