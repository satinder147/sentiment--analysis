import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical


from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


data=pd.read_csv('text_emotion.csv')
dict={'sadness': 0, 'enthusiasm': 1, 'neutral': 1, 'worry': 0, 'surprise': 2, 'love': 2, 'fun': 2, 'hate': 0, 'happiness': 2, 'boredom': 0, 'relief': 1, 'anger': 0, 'empty': 1}
data.sentiment=[dict[i] for i in data.sentiment]
sentiments=data.sentiment.values
content=data.content.values
vocab_size=25000
tk=Tokenizer(num_words=vocab_size)
tk.fit_on_texts(content)
content=tk.texts_to_sequences(content)

X = np.array(sequence.pad_sequences(content, maxlen=20, padding='post'))
y=sentiments
print(y.shape)
y=np.array(y)
y=to_categorical(y,num_classes=3)
'''
model = Sequential()

model.add(Embedding(vocab_size, 64, input_length=20))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(3,activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, batch_size=256, verbose=1, validation_split=0.2, epochs=20)

model.save('in.MODEL')
'''

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(25000, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
batch_size = 32
model.fit(X, y, epochs = 7, batch_size=batch_size, verbose = 2,validation_split=0.2)
print(model.summary())
mode.save('new.MODEL')





#print(content[0])
#print(content.shape)
