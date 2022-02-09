from matplotlib import projections
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_som.som import SOM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import DataScripts.data_interpolate as interpolate
import random
import pandas
import time
import os

def reduce(x):
    som = SOM(m=3, n=1, dim=40)
    #som.fit(x)
    som.fit(x, epochs=1, shuffle=True)
    transformed = som.transform(x)

    return transformed


def data():
    dataset = []
    sample_size = 40

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
    som = reduce(x)

    if not os.path.exists('results'):
        os.makedirs('results')
    
    ax = plt.axes(projection='3d')
    for point in som:
        ax.scatter3D(point[0],point[1],point[2])

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/somResults/som_{currentTime}')

    np.save(f'ikt590/reducedDims/som/{currentTime}', som)


if __name__ == "__main__":
    main()