from base64 import decode
from operator import mod
from statistics import mode
from sklearn import decomposition
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout

def encoder(input_size = 80, output_size = 3):
    model = Sequential()
    model.add(Input(shape=(input_size,)))
    model.add(Dense(input_size, activation='sigmoid'))
    model.add(Dense(input_size/2, activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))

    return model

def decoder(input_size = 80, output_size = 3):
    model = Sequential()
    model.add(Input(output_size))
    model.add(Dense(output_size, activation='sigmoid'))
    model.add(Dense(input_size/2, activation='relu'))
    model.add(Dense(input_size, activation='sigmoid'))
    
    return model

def lstm_encoder(input_size = 80, output_size = 3):
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(input_size, 1)))
    model.add(Dense(output_size, activation='sigmoid'))
    # model.add(LSTM(output_size, activation='sigmoid'))

    return model

def getModels(input_size = 80, output_size = 3):
    en = encoder(input_size, output_size)
    # en = lstm_encoder(input_size, output_size)
    de = decoder(input_size, output_size)

    return en, de