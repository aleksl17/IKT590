from matplotlib import projections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import random
import pandas
import time
import os

import helpers.data_manipulation as data_manipulation


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def reduce(xRed, n_components = 2):
        pca = PCA(n_components)
        pca = pca.fit(xRed)
        pc = pca.transform(xRed)
        return pc

    if not os.path.exists('reducedDims/pca'):
        os.makedirs('reducedDims/pca')
    
    dims = 3

    currentTime = str(int(time.time()))
    meta, x = data_manipulation.read_dataset(datasetFile='./dataset/dataset.json')
    x = random.sample(x, 10000)
    logger.debug(x)
    x = StandardScaler().fit_transform(x)
    pc = reduce(x, n_components=dims)

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/pcaResults'):
        os.makedirs('results/pcaResults')
    if not os.path.exists('reducedDims/pca'):
        os.makedirs('reducedDims/pca')

    if dims == 2:
        for point in pc:
            plt.scatter(point[0],point[1])
        
        plt.savefig(f'results/pcaResults/pca_{currentTime}')
    if dims == 3:
        ax = plt.axes(projection='3d')
        for point in pc:
            ax.scatter3D(point[0],point[1],point[2])
        
        plt.savefig(f'results/pcaResults/pca_{currentTime}')

    np.save(f'reducedDims/pca/{currentTime}', pc)


if __name__ == "__main__":
    main()
