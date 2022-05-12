import matplotlib.pyplot as plt
import numpy as np
import logging
import random
import numpy
import time
import os

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from helpers.performance import performance_for_algorithm


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Variables
    version='2.0'
    
    def dbscan(x, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        return dbscan.fit_predict(x), dbscan # Returns -1 if sample is "noisy". See documentation.
    
    def elbow_method(x, reduction, eps_min=0.1, min_samples_min=1, eps_max=1, min_samples_max=10, figDir='./.figs/'):
        scores = []
        while eps_min < eps_max:
            while min_samples_min < min_samples_max:
                dbscan_pred, dbsscan_model = dbscan(x, eps_min, min_samples_min)
                score = silhouette_score(x, dbscan_pred)
                scores.append(score)
                min_samples_min += 1
                print(f"min samples: {min_samples_min}")
            eps_min += 0.1
            print(f"eps min: {eps_min}")
        
        plt.clf()
        plt.plot(scores)
        plt.savefig(os.path.join(figDir + "elbow_method-" + reduction + "-" + currentTime))
        plt.clf()
    
    
    def makeFig(x, y, reduction, figDir='./.figs/'):
        # Color palette made with: https://mokole.com/palette.html
        # Settings: 16, 5%, 90%, 15000. Result: Perceived distance of 50.84, 46233 loops
        colors = ['darkslategray','maroon', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']
        
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=colors[0])
        ax.legend(['0'])
        plt.title(f'DBSCAN on {reduction}')
        plt.savefig(os.path.join(figDir + "DBSCAN-" + reduction + '-' + currentTime))
        plt.clf()

    
    if not os.path.exists('./results/dbscan'):
        os.makedirs('./results/dbscan')

    # PCA
    # logging.info('Staging PCA')
    # print('Staging PCA')
    # xPCA = np.load(os.path.join('reducedDims/pca/V' + version + '/pca.npy'))
    # numpy.random.shuffle(xPCA)
    # xPCA = xPCA[0:1000]
    # PCA_pred = dbscan(xPCA)
    # makeFig(xPCA, PCA_pred, 'pca')
    # np.save(os.path.join('results/dbscan/dbscan_pca_' + currentTime), PCA_pred)
    # PCA_pred = elbow_method(xPCA, "xPCA")
    
    # AE
    # logging.info('Staging AE')
    # print('Staging AE')
    # xAE = np.load(os.path.join('reducedDims/autoencoder/V' + version + '/autoencoder.npy'))
    # numpy.random.shuffle(xAE)
    # xAE = xAE[:1000]
    # AE_pred = dbscan(xAE)
    # makeFig(xAE, AE_pred, 'ae')
    # np.save(os.path.join('results/dbscan/dbscan_ae_' + currentTime), AE_pred)
    # AE_pred = elbow_method(xAE, "xAE")

    # SOM
    logging.info('Staging SOM')
    print('Staging SOM')
    xSOM = np.load(os.path.join('reducedDims/som/V' + version + '/som.npy'))
    numpy.random.shuffle(xSOM)
    xSOM = xSOM[:1000]
    # SOM_pred  = dbscan(xSOM)
    # makeFig(xSOM, SOM_pred, 'som')
    # np.save(os.path.join('results/dbscan/dbscan_som_' + currentTime), SOM_pred)
    SOM_pred = elbow_method(xSOM, "xSOM")

    # Performance
    # performance = performance_for_algorithm('DBSCAN', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)
    # np.save(os.path.join('results/dbscan/' + currentTime), performance)


if __name__ == "__main__":
    main()
