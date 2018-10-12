from keras.preprocessing.text import Tokenizer
import pandas as  pd


class tok:
    def __init__(self,name):
        self.data=pd.read_csv(name)
        self.content=self.data.content
    def tokenize(self,size):
        tk=Tokenizer(num_words=size)
        tk.fit_on_texts(self.content)
        return tk
