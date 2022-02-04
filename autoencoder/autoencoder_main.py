# from DataScripts.interpolate import interpolation
from operator import mod
from turtle import color
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from models import getModels
import numpy as np
import random
import pandas
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import DataScripts

#get data
dataset = []
sample_size = 40

for filename in os.listdir('./.localData'):
    csvData = pandas.read_csv(f"./.localData/{filename}")
    data = DataScripts.interpolation(csvData)
    for i in range(len(data)-sample_size):
        dataset.append(data[i:i + sample_size])


dataset = random.sample(dataset, 1000)
dataset = np.asarray(dataset)

x = []
# x = x/10 #wonky normalize

for d in dataset:
    minVal = min(d)*0.9
    x.append((d - minVal)/(max(d)-minVal))
    # x.append(d/max(d))


x = np.array(x)

#define models
encoder, decoder = getModels(sample_size, 3)
encoder._name = "encoder"
decoder._name = "decoder"

model = Sequential()
model.add(encoder)
model.add(decoder)

model.compile(optimizer='adam', loss='mse')

history = model.fit(x.reshape(x.shape[0], x.shape[1], 1),x, epochs=10000, shuffle=True)
# history = model.fit(x,x, epochs=10000, shuffle=True)

plt.plot(history.history['loss'])
plt.show()

x_test = x[:10]
pred = model.predict(x_test)

for i in range(len(x_test)):
    plt.plot(x_test[i], color='r')
    plt.plot(pred[i], color='b')
    plt.show()

model.save('autoencoder/model')
encoder.save('autoencoder/encoder')
