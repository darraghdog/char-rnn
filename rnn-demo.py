# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:49:54 2017

@author: dhanley2
ripped from @racheltho
https://github.com/fastai/courses/blob/a58e916834e3f0adb2b91b8d12f5dc2f6711a008/deeplearning1/nbs/lesson6.ipynb
"""
import os
print(os.getcwd())
# os.chdir('C:\\Users\\dhanley2\\Documents\\Projects\\demostt\\books')
from theano.sandbox import cuda
import utils; reload(utils)
from utils import *
from keras.layers import TimeDistributed, Activation
from numpy.random import choice
from keras.utils.data_utils import get_file


# path = get_file('data/dubliners.txt')
# text = open(path).read().lower()

with open('data/dubliners.txt', 'r') as myfile:
  text=myfile.read().replace('\n', ' ').replace('\r', ' ').lower()

print('corpus length:', len(text))

# Get array of total characters

chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)
chars.insert(0, "\0")
''.join(chars[1:-6])

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# convert the text to indices
idx = [char_indices[c] for c in text]
print(idx[:20])
print(''.join(indices_char[i] for i in idx[:70]))

# Preprocess and create model 
maxlen = 40
sentences = []
next_chars = []
for i in range(0, len(idx) - maxlen+1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i+1: i+maxlen+1])
print('nb sequences:', len(sentences))


sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])

print(sentences.shape, next_chars.shape)
n_fac = 24

model=Sequential([
        Embedding(vocab_size, n_fac, input_length=maxlen),
        LSTM(512, input_dim=n_fac,return_sequences=True, dropout_U=0.2, dropout_W=0.2,
             consume_less='gpu'),
        Dropout(0.2),
        LSTM(512, return_sequences=True, dropout_U=0.2, dropout_W=0.2,
             consume_less='gpu'),
        Dropout(0.2),
        TimeDistributed(Dense(vocab_size)),
        Activation('softmax')
    ])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())

def print_example():
    seed_string="he stood on the road and stared long at "
    for i in range(320):
        x=np.array([char_indices[c] for c in seed_string[-40:]])[np.newaxis,:]
        preds = model.predict(x, verbose=0)[0][-1]
        preds = preds/np.sum(preds)
        next_char = choice(chars, p=preds)
        seed_string = seed_string + next_char
    print(seed_string)

model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)
model.save_weights('weights/char_rnn_1.h5')
model.load_weights('weights/char_rnn_1.h5')
model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)
model.save_weights('weights/char_rnn_2.h5')
model.load_weights('weights/char_rnn_2.h5')
print_example()


model.optimizer.lr=0.0001
model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=256, nb_epoch=1)
model.save_weights('weights/char_rnn_3.h5')
model.load_weights('weights/char_rnn_3.h5')
print_example()

