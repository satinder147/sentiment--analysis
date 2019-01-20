from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from tokens import tok
import re

obj=tok('text_emotion.csv')
tk=obj.tokenize(25000)
model=load_model('one.MODEL')
print('model loaded')
dic={0:'negative',1:'neutral',2:'positive'}
a=pd.read_csv('validation_twitter.csv')
j=0;
k=0
while(j<93):
    #print('Enter your sentence or enter -1 to break')
    sentence=[]
    q=a['text'][j]
    q=re.sub('#','',q)
    q=re.sub('https\S+','',q)

    sentence.append(q)
    sentence=tk.texts_to_sequences(sentence)
    sentence = np.array(sequence.pad_sequences(sentence, maxlen=20, padding='post'))
    p=np.argmax(model.predict(sentence)[0])-1
    if(p==(a['sentiment'][j])):
        k=k+1
    #print(a['text'][j],a['sentiment'][j],p)
    j=j+1
    print(j)
print(k/93)
