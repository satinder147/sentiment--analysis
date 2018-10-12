import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tokens import tok
from mod import models
from keras.preprocessing import sequence
obj=models(3,25000,20)
ob=tok('text_emotion.csv')
voc=25000
tk=ob.tokenize(voc)


def preprocess():

    data=pd.read_csv('text_emotion.csv')
    dict={'sadness': 0, 'enthusiasm': 1, 'neutral': 1, 'worry': 0, 'surprise': 2, 'love': 2, 'fun': 2, 'hate': 0, 'happiness': 2, 'boredom': 0, 'relief': 1, 'anger': 0, 'empty': 1}
    data.sentiment=[dict[i] for i in data.sentiment]
    sentiments=data.sentiment.values
    content=data.content.values
    content=tk.texts_to_sequences(content)
    X = np.array(sequence.pad_sequences(content, maxlen=20, padding='post'))
    y=sentiments
    return X,y


X,y=preprocess()
y=np.array(y)
y=to_categorical(y,num_classes=3)
model=obj.arch1()
model.fit(X, y, batch_size=256, verbose=1, validation_split=0.2, epochs=20)
model.save('one.MODEL')
