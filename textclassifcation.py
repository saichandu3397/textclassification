import numpy as np 
import scipy
import csv
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt 
imdb_data=keras.datasets.imdb_data
(train_review,train_class),(test_review,test_class)=imdb_data.load_data(num_words=10000)
# num_wordds=10000 keeps the top 10000 most frequently used words.
x=len(train_class)
y=len(train_review)
# to get number of features
len(train_review[0])
train_review=keras.preprocessing.sequence.padsequences(train_review,value=word_index["<PAD>"],padding='post',maxlen=256)
test_review=keras.preprocessing.sequence.padsequences(test_review,value=word_index["<PAD>"],padding='post',maxlen=256)
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer=tf.train.AdamOptimizer(),loss='binary_crossentropy',metrics=['accuracy'])
#divide the train data into train and validation to avoid overfittiong
x_val=train_review(:10000)
partial_x_val=train_review(10000:)
y_val=train_class(:10000)
partial_y_val=train_class(10000:)
history=model.fit(partial_x_val,partial_y_val,epochs=40,batchsize=512,validation_data=(x_val,y_val),verbose=1)
results=model.evaluate(test_review,test_labels)
history_dict=history.history
history_dict.keys()