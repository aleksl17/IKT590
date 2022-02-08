from keras.models import load_model
import DataScripts.data_interpolate as interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import pandas
import time
import sys
import os

helper_folder = "ikt590/autoencoder"

def main():
    #get data
    dataset = []
    sample_size = 40

    for filename in os.listdir('./.localData'):
        csvData = pandas.read_csv(f"./.localData/{filename}")
        data = interpolate.interpolation(csvData)
        for i in range(len(data)-sample_size):
            dataset.append(data[i:i + sample_size])

    x = []
    dataset = random.sample(dataset, 100)
    dataset = np.asarray(dataset)
    for d in dataset:
        minVal = min(d)*0.9
        x.append((d - minVal)/(max(d)-minVal))
        # x.append(d/max(d))


    x = np.array(x)

    model = load_model(f"{helper_folder}/encoder")
    model.build()

    reduced_dims = model.predict(x)

    print(reduced_dims[:10])

    #save
    currentTime = str(int(time.time()))
    np.save(f'ikt590/reducedDims/autoencoder/{currentTime}', reduced_dims)

    ax = plt.axes(projection='3d')
    for point in reduced_dims:
            ax.scatter3D(point[0],point[1],point[2])

    plt.show()

if __name__ == "__main__":
    main()
