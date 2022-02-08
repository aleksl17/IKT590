from matplotlib import projections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import pandas
import time
import os

import DataScripts.data_interpolate as interpolate


def reduce(x, n_components = 2):
    pca = PCA(n_components)
    pca = pca.fit(x)
    pc = pca.transform(x)


    return pc

def data():
    dataset = []
    sample_size = 24 * 4

    for filename in os.listdir('./.localData'):
        csvData = pandas.read_csv(f"./.localData/{filename}")
        data = interpolate.interpolation(csvData)
        for i in range(len(data)-sample_size):
            dataset.append(data[i:i + sample_size])


    dataset = random.sample(dataset, 100)
    x = np.asarray(dataset)

    return x

def main():
    dims = 3

    currentTime = str(int(time.time()))
    x = data()

    x = StandardScaler().fit_transform(x)
    pc = reduce(x, n_components=dims)

    if not os.path.exists('results'):
            os.makedirs('results')

    if dims == 2:
        for point in pc:
            plt.scatter(point[0],point[1])

        plt.savefig(f'results/pca_{currentTime}')
    if dims == 3:
        ax = plt.axes(projection='3d')
        for point in pc:
            ax.scatter3D(point[0],point[1],point[2])

        if not os.path.exists('results'):
                os.makedirs('results')
        plt.savefig(f'results/pca_{currentTime}')

    if not os.path.exists('principle_components'):
            os.makedirs('principle_components')

    np.save(f'principle_components/{currentTime}', pc)


if __name__ == "__main__":
    main()