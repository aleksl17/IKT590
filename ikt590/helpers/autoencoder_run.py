# from DataScripts.interpolate import interpolation
from operator import mod
from turtle import color
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from models import getModels
import numpy as np
import random
import pandas
import sys
import os

# Disables GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_interpolate
import data_manipulation

batches = 10
batch_size = 1000
batch_epoch = 1000
load_old_models = True


def train_batch(x, model, epochs):
    x = random.sample(x,1000)


#get data
dataset = []
sample_size = 40

# for filename in os.listdir('./.localData'):
#     csvData = pandas.read_csv(f"./.localData/{filename}")
#     data = data_interpolate.interpolation(csvData)
#     for i in range(len(data)-sample_size):
#         dataset.append(data[i:i + sample_size])

meta, dataset = data_manipulation.read_dataset(datasetFile='./dataset/dataset.json')

dataset = random.sample(dataset, 10000)
dataset = np.asarray(dataset)

x = []
# x = x/10 #wonky normalize

for d in dataset:
    minVal = min(d)*0.9
    x.append((d - minVal)/(max(d)-minVal))
    # x.append(d/max(d))


# x = np.array(x)
if load_old_models:
    encoder = load_model('models/encoder')
    decoder = load_model('models/decoder')

else:    
    #define models
    encoder, decoder = getModels(sample_size, 3)
    encoder._name = "encoder"
    decoder._name = "decoder"

model = Sequential()
model.add(encoder)
model.add(decoder)
model.compile(optimizer='adam', loss='mse')

# history = model.fit(x.reshape(x.shape[0], x.shape[1], 1),x, epochs=10000, shuffle=True)
# history = model.fit(x,x, epochs=10000, shuffle=True)

loss_list = []

for i in range(batches):
    print('_________________________________________________________________')
    print(f'Training for batch: {i} of {batches}')
    print(f'Total training epochs: {i * batch_epoch}. Total training data: {i*batch_size}')
    x0 = np.array(random.sample(x,batch_size))
    history = model.fit(x0.reshape(x0.shape[0], x0.shape[1], 1),x0, epochs=batch_epoch, shuffle=True, verbose=0)
    loss = sum(history.history['loss']) / len(history.history['loss'])

    print(f'Loss for batch {i}: {loss }')
    loss_list.append(loss)

    print(f'Saving models for batch: {i}')
    # model.save('autoencoder/model')
    encoder.save('models/encoder')
    decoder.save('models/decoder')


plt.plot(loss_list)
plt.show()
plt.clf()
#train once on all
print("Training once on all data")
x = np.array(x)
history = model.fit(x.reshape(x.shape[0], x.shape[1], 1),x, epochs=batch_epoch, shuffle=True, verbose=0)
x_test = x[:10]
pred = model.predict(x_test)

for i in range(len(x_test)):
    plt.plot(x_test[i], color='r')
    plt.plot(pred[i], color='b')
    plt.show()
