import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
y = data.is_duplicate.values

# Keras text tokenizer
tk = text.Tokenizer(nb_words=200000)

max_len = 40

# Create a tokenizer fitted on all the words and fit the first and second sentence on it
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)
x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)
word_index = tk.word_index
print("Word index length : ", len(word_index))

# Make the target one-hot
ytrain_enc = np_utils.to_categorical(y)

"""
# define the size of the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
"""

print('Build model...')
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))


merged_model = Sequential()
merged_model.add(Merge([model5, model6], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())


merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('data/quora-weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit([x1, x2], y=y, batch_size=128, nb_epoch=200,
                 verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpoint])

#########################################
# Using only raw text
#########################################
# Epoch 4/200
# 323328/323432 [============================>.] - ETA: 0s - loss: 0.3505 - acc: 0.8388Epoch 00003: val_acc improved from 0.78183 to 0.79186, saving model to data/q323432/323432 [==============================] - 681s - loss: 0.3505 - acc: 0.8388 - val_loss: 0.4632 - val_acc: 0.7919
# Epoch 5/200
# 323432/323432 [==============================] - 616s - loss: 0.3083 - acc: 0.8593 - val_loss: 0.5002 - val_acc: 0.7884ve
# Epoch 6/200
# 323432/323432 [==============================] - 616s - loss: 0.2780 - acc: 0.8748 - val_loss: 0.4991 - val_acc: 0.7906ve
# Epoch 7/200
# 323328/323432 [============================>.] - ETA: 0s - loss: 0.2528 - acc: 0.8877Epoch 00006: val_acc improved from 0.79186 to 0.79443, saving model to data/q323432/323432 [==============================] - 676s - loss: 0.2528 - acc: 0.8877 - val_loss: 0.5045 - val_acc: 0.7944

