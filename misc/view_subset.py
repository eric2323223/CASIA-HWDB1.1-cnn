#!/usr/bin/env python2
# This script is useful to check if the CASIA HWDB1.1 subset was created correctly
import random
import sys

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 2:
    print('Usage: %s subset_filepath' % sys.argv[0])
    exit()

subset_filepath = sys.argv[1]
dataset = h5py.File(subset_filepath, 'r')

def showContent(f):
    for key in f.keys():
        print(f.get(key))


def showRandomImage(f):
    while True:
        dset = random.choice(['trn', 'tst'])
        i = random.randint(0, len(f[dset+'/x']))

        bitmap = f[dset+'/x'][i]
        print(np.where(f[dset+'/y'][i]==1))
        # print(bitmap)
        print(bitmap.shape)
        # print(np.mean(bitmap))
        assert sum(f[dset+'/y'][i]) == 1

        plt.imshow(np.squeeze(bitmap, axis=0), cmap=cm.Greys_r)
        plt.show()

def showCharactors(f, cidx):
    # print(f['vld/y'])
    for i in range(len(f['trn/y'])):
        index = (np.where(f['trn/y'][i]==1)[0][0])
        if cidx == index:
            print(i)
            print(f['trn/y'][i])
            img = f['trn/x'][i]
            plt.imshow(np.squeeze(img, axis=0), cmap=cm.Greys_r)
            plt.show()

def rand_samples(f, n):
    for i in range(n):
        idx = random.randint(0,len(f['trn/x']))
        print(np.where(f['trn/y'][idx]==1)[0][0])
        img = f['trn/x'][idx]
        plt.imshow(np.squeeze(img, axis=0), cmap=cm.Greys_r)
        plt.show()



# showContent(dataset)
showCharactors(dataset, 43)
# rand_samples(dataset, 50)
