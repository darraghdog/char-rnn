# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:49:54 2017

@author: dhanley2
"""
# https://github.com/fastai/courses/blob/a58e916834e3f0adb2b91b8d12f5dc2f6711a008/deeplearning1/nbs/lesson6.ipynb
import os
print(os.getcwd())
os.chdir('C:\\Users\\dhanley2\\Documents\\Projects\\demostt\\books')
#from theano.sandbox import cuda
#import utils; reload(utils)
#from utils import *
#from __future__ import division, print_function

#from keras.layers import TimeDistributed, Activation
#from numpy.random import choice

path = get_file('dubliners.txt', origin="C:\\Users\\dhanley2\\Documents\\Projects\\demostt\\books")
text = open(path).read().lower()
print('corpus length:', len(text))

# Get array of total characters

from keras.utils.data_utils import get_file
chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)
chars.insert(0, "\0")
''.join(chars[1:-6])

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

