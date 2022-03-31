from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import helpers.data_manipulation as data_manipulation
from performance import performance_for_algorithm
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
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

    def dbscan(x, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # x0 = random.sample(x, 1000)
        # dbscan.fit(x0)
        return dbscan.fit_predict(x)
    
    def cluster(x, reduction, eps=0.5, min_samples=5, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        dbscan_pred = dbscan(x)
        k = max(dbscan_pred)

        colors = ['blue','red', 'yellow', 'green', 'fuchsia', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']
        #LDA

        if k < 2:
            return dbscan_pred
        lda = LDA(n_components=2)
        lda_transformed = pd.DataFrame(lda.fit_transform(x, dbscan_pred))
        print(lda_transformed)

        plt.clf()
        fig, ax = plt.subplots(1)
        wi, hi = fig.get_size_inches()
        fig.set_size_inches(wi,hi)

        for i in range(k):
            ax.scatter(lda_transformed[dbscan_pred == i][0], lda_transformed[dbscan_pred == i][1], color=colors[i])
        # scatter = ax.scatter3D(x,color=colors[kmeans_pred])
        ax.legend(list(range(k)))
        logger.debug("Loading Figure")
        plt.title(f'DBSCAN on {reduction}')
        plt.savefig(os.path.join(figDir + "DBSCAN-" + reduction + '-' + currentTime))
        # plt.show()

        return dbscan_pred

    meta, dataset = data_manipulation.read_dataset(datasetFile='./.dataset/dataset.json')


    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/1648634711.npy').tolist()
    tmpData, xPCA = zip(*random.sample(list(zip(dataset, xPCA)),10000))

    PCA_pred = cluster(xPCA, 'pca', 0.1, 7)

    k = max(PCA_pred) + 1

    plt.clf()
    fig, axs = plt.subplots(k)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi,hi*k/2)
    fig.suptitle('clusters')

    for i, ax in enumerate(axs):
        ax.set_title("Cluster: " + str(i))

    for c, s in zip(PCA_pred, tmpData):
        axs[c].plot(s, color='black', alpha=.2)
    
    fig.tight_layout()
    plt.savefig(os.path.join('./.figs/' + "DBSCAN-PCA-real"  + currentTime))
    # plt.show()
    print("PCA DONE")


    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/1648632780.npy').tolist()

    tmpData, xAE = zip(*random.sample(list(zip(dataset, xAE)),10000))

    AE_pred = cluster(xAE, 'autoencoder', 0.001, 7)
    k = max(AE_pred) + 1

    if k > 1:

        plt.clf()
        fig, axs = plt.subplots(k)
        wi, hi = fig.get_size_inches()
        fig.set_size_inches(wi,hi*k/2)
        fig.suptitle('clusters')

        for i, ax in enumerate(axs):
            ax.set_title("Cluster: " + str(i))

        for c, s in zip(AE_pred, tmpData):
            axs[c].plot(s, color='black', alpha=.2)
        
        fig.tight_layout()
        plt.savefig(os.path.join('./.figs/' + "DBSCAN-AE-real"  + currentTime))
        # plt.show()
    print("AE DONE")

    #SOM
    logging.info('Staging SOM')
    xSOM = np.load('reducedDims/som/1648633743.npy').tolist()

    tmpData, xSOM = zip(*random.sample(list(zip(dataset, xSOM)),10000))
    
    SOM_pred  = cluster(xSOM, 'SOM', 0.0001, 4)
    k = max(SOM_pred) + 1

    if k > 1:
        plt.clf()
        fig, axs = plt.subplots(k)
        wi, hi = fig.get_size_inches()
        fig.set_size_inches(wi,hi*k/2)
        fig.suptitle('clusters')

        for i, ax in enumerate(axs):
            ax.set_title("Cluster: " + str(i))

        for c, s in zip(SOM_pred, tmpData):
            axs[c].plot(s, color='black', alpha=.2)
        
        fig.tight_layout()
        plt.savefig(os.path.join('./.figs/' + "DBSCAN-SOM-real"  + currentTime))
        # plt.show()
    print("SOM DONE")
    
    #performance
    performance = performance_for_algorithm('DBSCAN', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)


if __name__ == "__main__":
    main()
