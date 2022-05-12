from turtle import color
from performance import performance_for_algorithm
import helpers.data_manipulation as data_manipulation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import time
import os


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
            kmeans.fit(x)
            return kmeans.predict(x)
    
    def cluster(x, reduction, k = 3, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        kmeans_pred = kMeans(x,k, False)

        colors = ['lime', 'orange','darkslategray','maroon', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime']

        ax = plt.axes(projection='3d')
        for point, c in zip(x, kmeans_pred):
            ax.scatter3D(point[0], point[1], point[2], color=colors[c], label=c)
        
        logger.debug("Loading Figure")
        plt.title(f'K-Means on {reduction}')
        plt.savefig(os.path.join(figDir + "KMeans-" + reduction + '-' + currentTime))
        plt.clf()

        return kmeans_pred
    


    meta, dataset = data_manipulation.read_dataset(datasetFile='./.dataset/dataset.json')

    #PCA
    # logging.info('Staging PCA')
    # xPCA = np.load('reducedDims/pca/1644398105.npy').tolist()
    # xPCA = random.sample(xPCA, 10000)
    # PCA_pred = cluster(xPCA, 'pca', k=3)
    
    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/1648023850.npy').tolist()

    tmpData, xAE = zip(*random.sample(list(zip(dataset, xAE)),1000))

    print(len(tmpData))

    print(len(xAE))
    # xAE = random.sample(xAE, 10000)
    AE_pred = cluster(xAE, 'autoencoder', k=10)

    plt.clf()
    fig, axs = plt.subplots(10)
    fig.suptitle('clusters')

    for i, ax in enumerate(axs):
        ax.set_title("Cluster: " + str(i))

    for c, s in zip(AE_pred, tmpData):
        axs[c].plot(s)
    
    fig.tight_layout()
    # plt.savefig(os.path.join('./.figs/' + "KMeans-AE-real"  + currentTime))
    plt.show()
        

    #SOM
    # logging.info('Staging SOM')
    # xSOM = np.load('reducedDims/som/1644406419.npy').tolist()
    # xSOM = random.sample(xSOM, 10000)
    # SOM_pred  = cluster(xSOM, 'SOM', k=3)
    
    # performance = performance_for_algorithm('KMeans', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)

if __name__ == "__main__":
    main()
