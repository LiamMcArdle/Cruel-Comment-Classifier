import numpy as np 
import pandas as pd 
import os, sys, re, csv, codecs

import matplotlib.pyplot as plt 

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

train = pd.read_csv('train/train.csv', encoding = 'ascii', encoding_errors='backslashreplace')
test = pd.read_csv('test/test.csv', encoding = 'ascii', encoding_errors='backslashreplace')

classifications = ["Toxic", "Extremely Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
Y = train[classifications].values

train_sentences = train["comment_text"]
test_sentences = test["comment_text"]

#we will feed the comments into an LSTM (Long Short Term Memory) network. There are some steps we need to do:
#1. Tokenization - turn each sentence into a list of words 
#2. Indexing - index each word in a dictionary 
#3. Index Representation - Feed the indexed representations to the LSTM 

max_indexes = 25000
tokenizer = Tokenizer(num_words=max_indexes)
tokenizer.fit_on_texts(list(train_sentences))
tokenized_train = tokenizer.texts_to_sequences(train_sentences)
tokenized_test = tokenizer.texts_to_sequences(test_sentences)

#add padding to limit any abnormally long sentences, effectively saving memory and time at a small cost of model accuracy 

max_length = 150
X_train = pad_sequences(tokenized_train, maxlen=max_length)
X_test = pad_sequences(tokenized_test, maxlen=max_length)

#Input layer
inpt = Input(shape = (max_length,   ))

#Embedding layer - list of coordinates to words in vector space
embed_size = 200
X = Embedding(max_length, embed_size)(inpt)

#LSTM layer 
X = LSTM(80, return_sequences=True, name='LSTM_layer')(X) 

#Pooling to reshape 3D tensor to 2D
X = GlobalMaxPool1D()(X)

#Dropout layer
X = Dropout(0.1)(X)

X = Dense(50, activation="relu")(X)

#Dropout again
X = Dropout(0.1)(X)

#Sigmoid layer - for binary classification
X = Dense(6, activation="sigmoid")(X)

#Define inputs, outputs and configure learning process
model = Model(inputs=inpt, outputs=X)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 3
predictions = model.fit(X_train, Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save("cruelcommentclassifier_model.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), predictions.history['loss'], label="train_loss")
plt.plot(np.arange(0, epochs), predictions.history['val_loss'], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
