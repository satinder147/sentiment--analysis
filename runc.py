from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from tokens import tok
import re
from nltk.corpus import stopwords
sw=stopwords.words('english')

#from int import r
def r1(name):
    global sw	
    data=pd.read_csv(name,encoding='ISO-8859-1',skipinitialspace=True)
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
    return data


def r2(name):
    global sw
    data=pd.read_csv(name,skipinitialspace=True)
    #print(data["text"][3])
    data["text"]=data["text"].str.lower()
    print("lowercased")

    data["text"]=data["text"].apply(lambda x:re.sub(r'[@#]\w*','',x))  #removal of hastags
    print("remove hastags")
    data["text"]=data["text"].apply(lambda x:re.sub("\d+", "", x))
    data["text"]=data["text"].apply(lambda x: re.sub(r'[^\w]', ' ', x)) #removal of punctuations
    print("removed punctuation")

    data["text"]=data["text"].apply(lambda x: re.sub(r'https:\/\/.*','',x)) #removal of urls
    print("removed urls")


    for i in data["text"]:
        for j in sw:
            i=re.sub(j,'',i)
    print("removed stopwords")

    data["text"]=data["text"].apply(lambda x:re.sub(r' +',' ',x))
    print("extra spaces removed")


    for i in data["text"]:
        i=i.strip()
    print("leading and ending spaces removed") ## lstrip,rstrip
    return data



data=r1("train.csv")

obj=tok(data)
tk=obj.tokenize(25000)
model=load_model('newest.MODEL')
print('model loaded')
dic={0:'negative',1:'positive'}
a=r2("validation_twitter.csv")
j=0;
k=0
while(j<93):
    #print('Enter your sentence or enter -1 to break')
    sentence=[]
    q=a['text'][j]

    sentence.append(q)
    sentence=tk.texts_to_sequences(sentence)
    sentence = np.array(sequence.pad_sequences(sentence, maxlen=20, padding='post'))
    p=np.argmax(model.predict(sentence)[0])-1
    if(p==(a['sentiment.1'][j])):
        k=k+1
    #print(a['text'][j],a['sentiment'][j],p)
    j=j+1
    print(j)
print(k/93)
