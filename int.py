import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tokens import tok
from mod import models
from keras.preprocessing import sequence
import re
from nltk.corpus import stopwords
sw=stopwords.words('english')
obj=models(2,25000,21)
ob=tok('train.csv')
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

def r(data):
    #print(data["SentimentText"][3])
    data["SentimentText"]=data["SentimentText"].str.lower()
    print("lowercased")

    data["SentimentText"]=data["SentimentText"].apply(lambda x:re.sub(r'[@#]\w*','',x))  #removal of hastags
    print("remove hastags")
    data["SentimentText"]=data["SentimentText"].apply(lambda x:re.sub("\d+", "", x))
    data["SentimentText"]=data["SentimentText"].apply(lambda x: re.sub(r'[^\w]', ' ', x)) #removal of punctuations
    print("removed punctuation")

    data["SentimentText"]=data["SentimentText"].apply(lambda x: re.sub(r'https:\/\/.*','',x)) #removal of urls
    print("removed urls")


    for i in data["SentimentText"]:
        for j in sw:
            i=re.sub(j,'',i)
    print("removed stopwords")

    data["SentimentText"]=data["SentimentText"].apply(lambda x:re.sub(r' +',' ',x))
    print("extra spaces removed")


    for i in data["SentimentText"]:
        i=i.strip()
    print("leading and ending spaces removed") ## lstrip,rstrip

    content=data.SentimentText.values
    sentiment=data.Sentiment.values
    content=tk.texts_to_sequences(content)
    X = np.array(sequence.pad_sequences(content, maxlen=20, padding='post'))
    y=sentiments
    return X,y



def pre():
    data=pd.read_csv('train.csv',encoding='ISO-8859-1',skipinitialspace=True)
    X,y=r(data)
    return X,y

X,y=pre()
y=np.array(y)
y=to_categorical(y,num_classes=2)
model=obj.arch2()
model.fit(X, y, batch_size=1024, verbose=1, validation_split=0.2, epochs=20)
model.save('newest.MODEL')
