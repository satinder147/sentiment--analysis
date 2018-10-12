from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,SpatialDropout1D
from keras.layers import Embedding
class models:
    def __init__(self,out,voc,in_len):
        self.out=out
        self.voc=voc
        self.in_len=in_len

    def arch1(self):
        lstm_out = 196
        model = Sequential()
        model.add(Embedding(self.voc, 128,input_length = self.in_len))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(self.out,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        return model
