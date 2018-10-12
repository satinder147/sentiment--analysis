from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
model=load_model('in.MODEL')
#model.summary()
import pandas as pd

data=pd.read_csv('text_emotion.csv')
content=data.content.values
vocab_size=25000
tk=Tokenizer(num_words=vocab_size)
tk.fit_on_texts(content)






sentence=['I have completed my work']

sentence=tk.texts_to_sequences(sentence)
sentence = np.array(sequence.pad_sequences(sentence, maxlen=20, padding='post'))
#print(sentence)
print(sentence.shape)
print('ans')
print(np.argmax(model.predict(sentence)[0]))
