from matplotlib import projections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_som.som import SOM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import helpers.data_interpolate as interpolate
import helpers.data_manipulation as data_manipulation
import random
import pandas
import time
import os
import logging
import time


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    def reduce(xRed):
        som = SOM(m=3, n=1, dim=40)
        #som.fit(x)
        som.fit(xRed, epochs=10, shuffle=False)
        transformed = som.transform(xRed)
        return transformed
    
    dims = 3

    meta, dataset = data_manipulation.read_dataset(datasetFile='./datasets/dataset.json')
    # dataset = random.sample(dataset, 10000)
    x = np.asarray(dataset)

    currentTime = str(int(time.time()))
    # x = data()

    x = StandardScaler().fit_transform(x)
    print("Before reduce")
    logging.debug("Before reduce")
    som = reduce(x)
    print("After reduce")
    logging.debug("After reduce")
    
    # ax = plt.axes(projection='3d')
    # for point in som:
    #     ax.scatter3D(point[0],point[1],point[2])

    # if not os.path.exists('results'):
    #     os.makedirs('results')
    # plt.savefig(f'results/som_{currentTime}')

    if not os.path.exists('reducedDims/som'):
        os.makedirs('reducedDims/som')
    np.save(f'reducedDims/som/{currentTime}', som)
    print("Done!")


if __name__ == "__main__":
    main()