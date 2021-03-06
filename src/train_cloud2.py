#!/usr/bin/env python2
import json
import sys
import time

import numpy as np
import argparse
import tensorflow as tf
import io
import scipy.io as sio

np.random.seed(1337)

import h5py
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2

from tensorflow.python.lib.io import file_io

from keras import backend as K
from os import environ
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# user defined function to change keras backend
def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


def main(_):
    # if len(sys.argv) != 2:
    #     print('Usage: %s subset_filepath' % sys.argv[0])
    #     sys.exit(1)

    weightsTrainX = os.path.join(FLAGS.buckets, "trainx.npz")
    weightsTrainY = os.path.join(FLAGS.buckets, "trainy.npz")
    # modelWeightsX = os.path.join(FLAGS.buckets, "trainx.npz")
    # modelWeightsX = os.path.join(FLAGS.buckets, "trainx.npz")

    # call the function with "theano"
    # set_keras_backend("tensorflow")

    # subset_filepath = sys.argv[1]

    # as described in http://yuhao.im/files/Zhang_CNNChar.pdf
    model = Sequential()
    model.add(Conv2D(64, (3, 3),
                     activation='relu', padding='same', strides=(1, 1),
                     input_shape=(64, 64, 1)))
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
    # with open('%d-model.json' % timestamp, 'w') as f:
    #     d = json.loads(model.to_json())
    #     json.dump(d, f, indent=4)

    # contentString = file_io.read_file_to_string(weightsTrainX, binary_mode=False)
    # contentString = np.load(file_io.FileIO(weightsTrainX, 'r'))
    f = io.BytesIO(file_io.read_file_to_string(weightsTrainX, binary_mode=True))
    data = np.load(f)
    print(data['arr_0'].shape)

    f = io.BytesIO(file_io.read_file_to_string(weightsTrainY, binary_mode=True))
    data = np.load(f)
    print(data['arr_0'].shape)

    # f = io.StringIO(unicode(contentString, 'utf-8'))

    # model.fit(converted_trn, f['trn/y'], validation_data=(converted_vld, f['vld/y']),
    #           epochs=15, batch_size=128, shuffle='batch', verbose=1)
    #
    # score = model.evaluate(converted_tst, f['tst/y'], verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    #
    # model.save_weights('%d-weights-%f.hdf5' % (timestamp, score[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)