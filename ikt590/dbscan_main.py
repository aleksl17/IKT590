import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os

from sklearn.cluster import DBSCAN

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

        return dbscan.fit_predict(x) # Returns -1 if sample is "noisy". See documentation.
    
    
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
    logging.info('Staging PCA')
    print('Staging PCA')
    xPCA = np.load(os.path.join('reducedDims/pca/V' + version + '/pca.npy'))
    # xPCA = random.sample(xPCA, 10000)
    PCA_pred = dbscan(xPCA, 0.3, 3)
    makeFig(xPCA, PCA_pred, 'pca')
    np.save(os.path.join('results/dbscan/dbscan_pca_' + currentTime), PCA_pred)
    
    # Autoencoder
    logging.info('Staging AE')
    print('Staging AE')
    xAE = np.load(os.path.join('reducedDims/autoencoder/V' + version + '/autoencoder.npy'))
    # xAE = random.sample(xAE, 10000)
    AE_pred = dbscan(xAE, 0.3, 3)
    makeFig(xAE, AE_pred, 'ae')
    np.save(os.path.join('results/dbscan/dbscan_ae_' + currentTime), AE_pred)

    # # SOM
    logging.info('Staging SOM')
    print('Staging SOM')
    xSOM = np.load(os.path.join('reducedDims/som/V' + version + '/som.npy'))
    # xSOM = random.sample(xSOM, 10000)
    SOM_pred  = dbscan(xSOM, 0.3, 3)
    makeFig(xSOM, SOM_pred, 'som')
    np.save(os.path.join('results/dbscan/dbscan_som_' + currentTime), SOM_pred)

    # Performance
    # performance = performance_for_algorithm('DBSCAN', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)
    # np.save(os.path.join('results/dbscan/' + currentTime), performance)


if __name__ == "__main__":
    main()
