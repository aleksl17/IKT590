import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import time
import os

from sklearn.utils import shuffle


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
    
    def cluster(x, reduction, k = 3, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        kmeans_pred = kMeans(x,k, False)

        colors = ['lightcoral', 'red', 'darkred', 'chocolate', 'bisque', 'darkorange', 'gold', 'yellow', 'olive', 'darkgreen', 'lime', 'aquamarine', 'teal', 'cyan', 'lightblue', 'steelblue', 'navy', 'blue', 'indigo', 'violet', 'purple', 'crimson', 'pink']
        shuffle(colors)

        print(kmeans_pred)

        ax = plt.axes(projection='3d')
        for point, c in zip(x, kmeans_pred):
            ax.scatter3D(point[0], point[1], point[2], color=colors[c])
        
        logger.debug("Loading Figure")
        plt.title(f'K-Means on {reduction}')
        plt.savefig(os.path.join(figDir + "KMeans-K=" +str(k)+ '-' + reduction + '-' + currentTime))
        plt.clf()

        return silhouette_score(x, kmeans_pred)
    
    def elbow_method(x, reduction):
        max_k = 10
        scores = []
        for k in range(2,max_k):
            scores.append(cluster(x, reduction, k))

        plt.plot(scores)
        plt.show()
        plt.clf()
    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/1644398105.npy').tolist()
    xPCA = random.sample(xPCA, 10000)
    PCA_pred = elbow_method(xPCA, 'pca')
    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/1644405844.npy').tolist()
    xAE = random.sample(xAE, 10000)
    AE_pred = elbow_method(xAE, 'autoencoder')

    #SOM
    logging.info('Staging SOM')
    xSOM = np.load('reducedDims/som/1644406419.npy').tolist()
    xSOM = random.sample(xSOM, 10000)
    SOM_pred  = elbow_method(xSOM, 'SOM')
    

if __name__ == "__main__":
    main()
