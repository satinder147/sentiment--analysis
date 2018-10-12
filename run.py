from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from tokens import tok

obj=tok('text_emotion.csv')
tk=obj.tokenize(25000)
model=load_model('one.MODEL')
print('model loaded')
dic={0:'negative',1:'neutral',2:'positive'}


while(1):
    print('Enter your sentence or enter -1 to break')
    sentence=[]
    i=input()
    if(i=='-1'):
        break
    sentence.append(i)
    sentence=tk.texts_to_sequences(sentence)
    sentence = np.array(sequence.pad_sequences(sentence, maxlen=20, padding='post'))
    print(dic[np.argmax(model.predict(sentence)[0])])
