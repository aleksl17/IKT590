from matplotlib import projections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_som.som import SOM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scripts.data_interpolate as interpolate
import scripts.data_manipulation as data_manipulation
import random
import pandas
import time
import os


def main():

    def reduce(xRed):
        som = SOM(m=3, n=1, dim=40)
        #som.fit(x)
        som.fit(xRed, epochs=1, shuffle=True)
        transformed = som.transform(xRed)
        return transformed
    
    dims = 3

    meta, dataset = data_manipulation.read_dataset(datasetFile='./dataset/dataset.json')
    dataset = random.sample(dataset, 10000)
    x = np.asarray(dataset)

    currentTime = str(int(time.time()))
    # x = data()

    x = StandardScaler().fit_transform(x)
    som = reduce(x)

    if not os.path.exists('results'):
        os.makedirs('results')
    
    ax = plt.axes(projection='3d')
    for point in som:
        ax.scatter3D(point[0],point[1],point[2])

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/som_{currentTime}')

    np.save(f'reducedDims/som/{currentTime}', som)


if __name__ == "__main__":
    main()