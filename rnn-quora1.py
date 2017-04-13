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


# split punctualtion function
def split_punctuation(s):
    try:
        return re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", s)
    except:
        return s



data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
y = data.is_duplicate.values

# Add a set where punctuation is split out 
data['question1split'] = data['question1'].map(lambda x:split_punctuation(x))
data['question2split'] = data['question2'].map(lambda x:split_punctuation(x))


# Keras text tokenizer
tk = text.Tokenizer(nb_words=200000)

max_len = 40

# Create a tokenizer fitted on all the words and fit the first and second sentence on it
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)) + list(data.question1split.values) + list(data.question2split.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)
x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

x3 = tk.texts_to_sequences(data.question1split.values)
x3 = sequence.pad_sequences(x3, maxlen=max_len)
x4 = tk.texts_to_sequences(data.question2split.values.astype(str))
x4 = sequence.pad_sequences(x4, maxlen=max_len)


word_index = tk.word_index
print("Word index length : ", len(word_index))

# Make the target one-hot
ytrain_enc = np_utils.to_categorical(y)

# Load embeddings and save embeddings for words in the dataset 
embeddings_index = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Create embedding matrix using the word embeddings
max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build Word embedding LTSM model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

# Create a 1D convolution on the inputted embeddings

print('Build 1D convolutional model...')

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(300))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())


print('Build model embedding on raw words...')
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))


merged_model = Sequential()
# merged_model.add(Merge([model1, model2, model3, model4, model5, model6, model1, model2, model3, model4, model5, model6], mode='concat'))
merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))

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

# merged_model.fit([x1, x2, x1, x2, x1, x2, x1, x2, x1, x2, x1, x2], y=y, batch_size=128, nb_epoch=200,
#                  verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpoint])


merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=128, nb_epoch=200,
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



#########################################
# Using Embeddings and raw text text
#########################################
# Epoch 4/200
# 323432/323432 [==============================] - 619s - loss: 0.3228 - acc: 0.8543 - val_loss: 0.4253 - val_acc: 0.8060ve
# Epoch 5/200
# 323328/323432 [============================>.] - ETA: 0s - loss: 0.2833 - acc: 0.8743Epoch 00004: val_acc improved from 0.80691 to 0.80865, saving model to data/q323432/323432 [==============================] - 708s - loss: 0.2833 - acc: 0.8743 - val_loss: 0.4252 - val_acc: 0.8087
# Epoch 6/200
# 323328/323432 [============================>.] - ETA: 0s - loss: 0.2516 - acc: 0.8895Epoch 00005: val_acc improved from 0.80865 to 0.81155, saving model to data/q323432/323432 [==============================] - 711s - loss: 0.2516 - acc: 0.8895 - val_loss: 0.4387 - val_acc: 0.8115
# Epoch 7/200
# 323328/323432 [============================>.] - ETA: 0s - loss: 0.2270 - acc: 0.9016Epoch 00006: val_acc improved from 0.81155 to 0.81182, saving model to data/q323432/323432 [==============================] - 707s - loss: 0.2270 - acc: 0.9016 - val_loss: 0.4646 - val_acc: 0.8118
# Epoch 8/200
# 323328/323432 [============================>.] - ETA: 0s - loss: 0.2103 - acc: 0.9096Epoch 00007: val_acc improved from 0.81182 to 0.81685, saving model to data/q323432/323432 [==============================] - 705s - loss: 0.2103 - acc: 0.9096 - val_loss: 0.4690 - val_acc: 0.8169


#########################################
# Using Embeddings and 1D conv and raw text
#########################################

# Epoch 2/200
# 323432/323432 [==============================] - 697s - loss: 0.4206 - acc: 0.8005 - val_loss: 0.4260 - val_acc: 0.7899 0.78363 to 0.78992, saving model to data/quora-weights.h5
# Epoch 3/200
# 323432/323432 [==============================] - 632s - loss: 0.3604 - acc: 0.8347 - val_loss: 0.4094 - val_acc: 0.8064 0.78992 to 0.80641, saving model to data/quora-weights.h5
# Epoch 4/200
# 323432/323432 [==============================] - 639s - loss: 0.3115 - acc: 0.8600 - val_loss: 0.4193 - val_acc: 0.8074 0.80641 to 0.80742, saving model to data/quora-weights.h5
# Epoch 5/200

