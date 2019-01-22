import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tokens import tok
from mod import models
from keras.preprocessing import sequence
import re
from nltk.corpus import stopwords

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

def r(name):
    data=pd.read_csv(name,skipinitialspace=True)

    dict={'sadness': 0, 'enthusiasm': 1, 'neutral': 1, 'worry': 0, 'surprise': 2, 'love': 2, 'fun': 2, 'hate': 0, 'happiness': 2, 'boredom': 0, 'relief': 1, 'anger': 0, 'empty': 1}
    data.sentiment=[dict[i] for i in data.sentiment]
    #print(data["sentiment"][3])
    data["content"]=data["content"].str.lower()
    print("lowercased")

    data["content"]=data["content"].apply(lambda x:re.sub(r'[@#]\w*','',x))  #removal of hastags
    print("remove hastags")
    data["content"]=data["content"].apply(lambda x:re.sub("\d+", "", x))
    data["content"]=data["content"].apply(lambda x: re.sub(r'[^\w]', ' ', x)) #removal of punctuations
    print("removed punctuation")

    data["content"]=data["content"].apply(lambda x: re.sub(r'https:\/\/.*','',x)) #removal of urls
    print("removed urls")


    for i in data["content"]:
        for j in sw:
            i=re.sub(j,'',i)
    print("removed stopwords")

    data["content"]=data["content"].apply(lambda x:re.sub(r' +',' ',x))
    print("extra spaces removed")


    for i in data["content"]:
        i=i.strip()
    print("leading and ending spaces removed") ## lstrip,rstrip
    return data




def pre():
    global data

    content=data.content.values
    sentiment=data.sentiment.values
    content=tk.texts_to_sequences(content)
    X = np.array(sequence.pad_sequences(content, maxlen=20, padding='post'))
    y=sentiment
    return X,y

sw=stopwords.words('english')
obj=models(3,25000,20)
data=r("text_emotion.csv")
ob=tok(data)
voc=25000
tk=ob.tokenize(voc)

X,y=pre()
y=np.array(y)
y=to_categorical(y,num_classes=3)
model=obj.arch2()
model.fit(X, y, batch_size=1024, verbose=1, validation_split=0.2, epochs=20)
model.save('latest.MODEL')
