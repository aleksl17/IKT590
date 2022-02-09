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

import DataScripts.data_manipulation as data_manipulation


def reduce(x, n_components = 2):
    pca = PCA(n_components)
    pca = pca.fit(x)
    pc = pca.transform(x)
    return pc

# def data():
#     datasetList = []
#     # sample_size = 24 * 4
#     # for filename in os.listdir('./.localData'):
#     #     csvData = pandas.read_csv(f"./.localData/{filename}")
#     #     data = interpolate.interpolation(csvData)
#     #     for i in range(len(data)-sample_size):
#     #         dataset.append(data[i:i + sample_size])
#     x = []
#     meta = []
#     datasetList = data_manipulation.read_dataset(datasetFile='./.dataset/dataset-1644394453.json')
#     for sensorItem in datasetList:
#         #dataset.append(sensorItem[1]) #[1] sample values
#         for sample in sensorItem[1]: # [1] samples, [0] metadata
#             x.append(sample)
#     for sensorItem in datasetList:
#         for sample in sensorItem[1]:
#             x.append(sample)
#         for m in sensorItem[0]:
#             meta.append(m)
#     x, meta = GetData()
    # dataset = random.sample(dataset, 100)
    # logger.debug(dataset)
    # x = np.asarray(dataset)
    # return x

def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if not os.path.exists('reducedDims/pca'):
        os.makedirs('reducedDims/pca')
    
    dims = 3

    currentTime = str(int(time.time()))
    meta, x = data_manipulation.read_dataset(datasetFile='./.dataset/dataset-1644397269.json')
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
