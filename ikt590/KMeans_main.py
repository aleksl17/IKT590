from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import time
import os

import DataScripts.data_manipulation as data_manipulation


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def kMeans(x, k=3, all=False):
        kmeans = KMeans(n_clusters = k)
        # labels = kmeans.labels_
        # # [0,1,0,2,0,1,1]
        # pred = kmeans.predict(x)
        # # [1,0,1,2,,1111]
        if all:
            print("Labels Coming Here Soon TM")
            # kmeans.fit(x)
            # return kmeans.labels_
        if not all:
            x0 = random.sample(x, 1000)
            kmeans.fit(x0)
            return kmeans.predict(x)
    
    def cluster(x, algorithm, k = 3):
        logger.debug(f'Kmeans for {algorithm}')
        kmeans_pred = kMeans(x,k, False)

        colors = ['r','b','g']

        ax = plt.axes(projection='3d')
        for point, c in zip(x, kmeans_pred):
            ax.scatter3D(point[0], point[1], point[2], color=colors[c])
        
        logger.debug("Loading Figure")
        plt.title(f'K-Means on {algorithm}')
        plt.show()
    
    x = np.load('reducedDims/pca/1644398105.npy').tolist()
    x = random.sample(x, 10000)
    print(x)

    cluster(x, 'pca', k=3)
    
    
    print("Hello, World!")



if __name__ == "__main__":
    main()
