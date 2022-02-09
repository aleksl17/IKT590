import imp
from keras.models import load_model
# import DataScripts.data_interpolate as interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import pandas
import time
import sys
import os

# from ..DataScripts import data_manipulation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import DataScripts
import DataScripts.data_manipulation as data_manipulation


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import DataScripts

#get data
# dataset = []
# sample_size = 40

# for filename in os.listdir('./.localData'):
#     csvData = pandas.read_csv(f"./.localData/{filename}")
#     data = interpolate.interpolation(csvData)
#     for i in range(len(data)-sample_size):
#         dataset.append(data[i:i + sample_size])

meta, dataset = data_manipulation.read_dataset(datasetFile='./.dataset/dataset-1644397269.json')
x = []
# dataset = random.sample(dataset, 100)
dataset = np.asarray(dataset)
for d in dataset:
    minVal = min(d)*0.9
    x.append((d - minVal)/(max(d)-minVal))
    # x.append(d/max(d))


x = np.array(x)

# model = load_model(os.path.join('%USERPROFILE%/IKT590/ikt590/autoencoder/encoder'))
model = load_model('C:/Users/Aegir/IKT590/ikt590/autoencoder/encoder/')
print(model)
model.build()

reduced_dims = model.predict(x)

print(reduced_dims[:10])

#save
currentTime = str(int(time.time()))
np.save(f'reducedDims/autoencoder/{currentTime}', reduced_dims)

ax = plt.axes(projection='3d')
for point in reduced_dims:
        ax.scatter3D(point[0],point[1],point[2])

plt.show()