#!/usr/bin/env python2
import sys

import h5py

import utils



def gb2312_to_character(l):
    return bytes([int(l[:2], 16), int(l[2:], 16)]).decode('gb2312')

table = {}
for i, (bitmap, tagcode) in enumerate(utils.read_gnt_in_directory('../data/test2')):
    # dset_bitmap[i]  = utils.normalize_bitmap(bitmap)
    if tagcode in table:
        table[tagcode] = table[tagcode]+1
    else:
        table[tagcode] = 1
    # print("GB code: {}, chinese char:{}".format(hex(tagcode), gb2312_to_character(hex(tagcode)[2:])))

print(len(table))
print(table)
total_sample = 0
for key, value in table.items():
    total_sample = total_sample + value
print(total_sample)