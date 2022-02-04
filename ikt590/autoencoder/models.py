from base64 import decode
from operator import mod
from statistics import mode
from sklearn import decomposition
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, Input, Flatten

def encoder(input_size = 80, output_size = 3):
    model = Sequential()
    model.add(Input(shape=(input_size,)))
    model.add(Dense(input_size, activation='sigmoid'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))

    model.summary()

    return model

def decoder(input_size = 80, output_size = 3):
    model = Sequential()
    model.add(Input(output_size))
    model.add(Dense(output_size, activation='sigmoid'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(input_size, activation='sigmoid'))
    
    return model

def lstm_encoder(input_size = 80, output_size = 3):
    # input = Input(shape=(input_size,1))
    # model = LSTM(output_size)(input)

    model = Sequential()
    model.add(LSTM(20, input_shape=(input_size, 1), return_sequences=True))
    model.add(LSTM(output_size, activation='relu', return_sequences=True))
    model.add(Flatten())
    model.add(Dense(output_size, activation='sigmoid'))

    model.summary()

    return model

def lstm_decoder(input_size = 80, output_size = 3):
    # input = Input(shape=(output_size))
    # model = RepeatVector(input_size)(input)
    # model = LSTM(input_size, return_sequences=True)(model)

    model = Sequential()
    model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(output_size,1)))
    model.add(LSTM(input_size, return_sequences=True))

    return model


def getModels(input_size = 80, output_size = 3):
    # en = encoder(input_size, output_size)
    en = lstm_encoder(input_size, output_size)
    de = decoder(input_size, output_size)
    # de = lstm_decoder(input_size, output_size)

    return en, de