import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
(train_x,train_y),(test_x,test_y)=imdb.load_data(num_words=10000)
# print('review is :',train_x[5])
# print('label is :',train_y[5])
# vocab=imdb.get_word_index()
# print(vocab)

maxlen = 100
train_x =pad_sequences(train_x,maxlen=maxlen)
test_x =pad_sequences(test_x,maxlen=maxlen)
# model creation

model = Sequential()
model.add(Embedding(10000, 32))
model.add(Conv1D(32, 7, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())
history = model.fit(train_x,train_y,
epochs =10,
batch_size=128,
validation_split=0.2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs , acc, 'b', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure(figsize=(12,8))
plt.plot(epochs, loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()