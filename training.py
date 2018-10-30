# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:41:52 2018

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

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras import backend as K

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

lr = 0.01
momentum = 0.3
batch_size = 5
epochs = 50

#x_train = data['x_train_1']
#y_train = data['y_train_1']
#x_test = data['x_test_1']
#y_test = data['y_test_1']

x_train = x_train.reshape(x_train.shape[0], 128, 256, 1)
x_test = x_test.reshape(x_test.shape[0], 128, 256, 1)

y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)

input_shape = (np.size(x_train, 1), np.size(x_train, 2), 1)

# ==============================================================================
# Create deep network
# ==============================================================================
K.clear_session()

inputs = Input(input_shape)
conv0 = Conv2D(32, kernel_size=(11, 11), strides=5, activation='relu',
               input_shape=input_shape)(inputs)
conv1 = Conv2D(64, (3,3), activation='relu')(conv0)
pool0 = MaxPooling2D(pool_size=(2, 2))(conv1)
pool0 = Dropout(0.25)(pool0)
flatt0 = Flatten()(pool0)
dense0 = Dense(128, activation='relu')(flatt0)
#dense0 = Dropout(0.25)(dense0)
outputs = Dense(3, activation='softmax')(dense0)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=lr, momentum=momentum),
              metrics=['accuracy'])
# ==============================================================================

# ==============================================================================
# Training deep network
# ==============================================================================
start_time = time.time()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
#                    validation_split=validation_split)
#                    validation_data=(x_validation, y_validation))
                    validation_data=(x_test, y_test))

training_time = time.time() - start_time

score = model.evaluate(x_train,
                       y_train,
                       verbose=0)
#
print("\n--- Training time: %s seconds ---" % training_time)
print('Traning loss:', score[0])
print('Training accuracy:', score[1])
# ==============================================================================

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

model_name = data_set_name + '_' + 'epochs_' + str(epochs) + '_' + 'acc_' + str_acc + '.h5'
model_path = os.path.join('model', model_name)
#model.save(model_path)

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

#np.savez(dataset_name+'_'+str(epochs)+ '_' + str_acc, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,acc=acc, loss=loss, val_acc=val_acc, val_loss=val_loss)

#ut.plot_acc_loss(acc, val_acc, loss, val_loss)