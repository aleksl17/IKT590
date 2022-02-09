from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import logging
import random
import time
import os

# TODO
# Rework "colors" variable.


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    def dbscan(x):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        # x0 = random.sample(x, 1000)
        # dbscan.fit(x0)
        return dbscan.fit_predict(x)
    
    def cluster(x, reduction, k = 3, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        dbscan_pred = dbscan(x)

        # colors = ['r','b','g','y','m','c','k']
        colors = ['lightcoral', 'red', 'darkred', 'chocolate', 'bisque', 'darkorange', 'gold', 'yellow', 'olive', 'darkgreen', 'lime', 'aquamarine', 'teal', 'cyan', 'lightblue', 'steelblue', 'navy', 'blue', 'indigo', 'violet', 'purple', 'crimson', 'pink']
        random.shuffle(colors)

        ax = plt.axes(projection='3d')
        for point, c in zip(x, dbscan_pred):
            ax.scatter3D(point[0], point[1], point[2], c=colors[c])
        
        logger.debug("Loading Figure")
        plt.title(f'DBSCAN on {reduction}')
        plt.savefig(os.path.join(figDir + "DBSCAN-" + reduction + '-' + currentTime))
        plt.clf()

        return dbscan_pred

    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/1644398105.npy').tolist()
    xPCA = random.sample(xPCA, 10000)
    PCA_pred = cluster(xPCA, 'pca', k=3)
    
    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/1644405844.npy').tolist()
    xAE = random.sample(xAE, 10000)
    AE_pred = cluster(xAE, 'autoencoder', k=3)

    #SOM
    logging.info('Staging SOM')
    xSOM = np.load('reducedDims/som/1644406419.npy').tolist()
    xSOM = random.sample(xSOM, 10000)
    SOM_pred  = cluster(xSOM, 'SOM', k=3)


if __name__ == "__main__":
    main()
