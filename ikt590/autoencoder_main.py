from keras.models import load_model
import helpers.data_interpolate as interpolate
import helpers.data_manipulation as data_manipulation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import pandas
import time
import sys
import os
import logging

# Disables GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# helper_folder = "ikt590/autoencoder"

def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    #get data
    dataset = []
    sample_size = 40

    # for filename in os.listdir('./.localData'):
    #     csvData = pandas.read_csv(f"./.localData/{filename}")
    #     data = interpolate.interpolation(csvData)
    #     for i in range(len(data)-sample_size):
    #         dataset.append(data[i:i + sample_size])
    meta, dataset = data_manipulation.read_dataset(datasetFile='./datasets/V3.0/dataset.json')

    x = []
    # dataset = random.sample(dataset, 10000)
    dataset = np.asarray(dataset)
    for d in dataset:
        minVal = min(d)*0.9
        x.append((d - minVal)/(max(d)-minVal))
        # x.append(d/max(d))


    x = np.array(x)

    # model = load_model(f"{helper_folder}/encoder")
    model = load_model('models/V3.0/encoder')
    # model.build()
    # model.compile(run_eagerly=True)

    print("Before predict")
    logging.debug("Before predict")
    reduced_dims = model.predict(x)
    logging.debug("After predict")
    print("After predict")

    # print(reduced_dims[:10])

    #save
    currentTime = str(int(time.time()))
    np.save(f'reducedDims/autoencoder/{currentTime}', reduced_dims)
    print("Done!")
    logging.debug("Done!")

    # ax = plt.axes(projection='3d')
    # for point in reduced_dims:
    #         ax.scatter3D(point[0],point[1],point[2])

    # plt.show()

if __name__ == "__main__":
    main()
