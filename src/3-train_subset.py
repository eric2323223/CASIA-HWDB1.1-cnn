#!/usr/bin/env python2
import json
import sys
import time

import numpy as np
from random import randint
np.random.seed(1337)

import h5py
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2

from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback


from keras import backend as K
from os import environ
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# user defined function to change keras backend
def set_keras_backend(backend):
    if K.backend() != backend:
       environ['KERAS_BACKEND'] = backend
       reload(K)
       assert K.backend() == backend


if len(sys.argv) != 2:
    print('Usage: %s subset_filepath' % sys.argv[0])
    sys.exit(1)

# call the function with "theano"
set_keras_backend("tensorflow")

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

subset_filepath = sys.argv[1]

# as described in http://yuhao.im/files/Zhang_CNNChar.pdf
model = Sequential()
model.add(Conv2D(64, (3, 3),
                        activation='relu', padding='same', strides=(1, 1),
                        input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), weights=[np.random.normal(0, 0.01, size=(3, 3, 64, 128)), np.zeros(128)],
                        activation='relu', padding='same', strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), weights=[np.random.normal(0, 0.01, size=(3, 3, 128, 256)), np.zeros(256)],
                        activation='relu', padding='same', strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.summary()
model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

timestamp = int(time.time())

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')



# with open('%d-model.json' % timestamp, 'w') as f:
#     d = json.loads(model.to_json())
#     json.dump(d, f, indent=4)

with h5py.File(subset_filepath, 'r') as f:
    print(f['trn/x'].shape)
    # img = np.reshape(f['trn/x'], (40000, 64,64))
    # print(img.shape)
    # plt.imshow(img[0], cmap='gray')
    # plt.show()
    #
    # for i in range(5):
    #     print("{}: {}".format(i, np.moveaxis(f['trn/x'].value, 1, i).shape))
    converted_trn = np.moveaxis(f['trn/x'].value, 1, 3)[:2000]
    idx = randint(0, 200)
    img = converted_trn[idx]
    img = np.reshape(img, (64,64))
    plt.imshow(img, cmap='gray')
    plt.show()
    converted_vld = np.moveaxis(f['vld/x'].value, 1, 3)
    converted_tst = np.moveaxis(f['tst/x'].value, 1, 3)

    callbacks_list = [checkpoint, TestCallback((converted_tst, f['tst/y']))]

    # model.fit(converted_trn, f['trn/y'][0:2000], validation_data=(converted_vld, f['vld/y']),
    #           epochs=2, batch_size=64, shuffle='batch', verbose=1, callbacks=callbacks_list)

    model.fit(converted_trn, f['trn/y'][0:2000], validation_split=0.25,
              epochs=20, batch_size=64, shuffle='batch', verbose=1, callbacks=callbacks_list)

    score = model.evaluate(converted_tst, f['tst/y'], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights('%d-weights-%f.hdf5' % (timestamp, score[1]))
