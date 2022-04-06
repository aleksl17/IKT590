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

    def dbscan(x, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        return dbscan.fit_predict(x)
    
    def cluster(x, reduction, eps, min_samples):
        logger.debug(f'Kmeans for {reduction}')
        dbscan_pred = dbscan(x)

        return dbscan_pred
    
    def makeFig(x, y, reduction, figDir='./.figs/'):
        # Color palette made with: https://mokole.com/palette.html
        # Settings: 16, 5%, 90%, 15000. Result: Percieved distance of 50.84, 46233 loops
        colors = ['darkslategray','maroon', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']

        # ax = plt.axes(projection='3d')
        # for point, c in zip(x, y):
        #     ax.scatter3D(point[0], point[1], point[2], c=colors[c])

        ax = plt.axes(projection='3d')

        print(max(y))

        for i in range(max(y)+1):
            scatx = []
            scaty = []
            scatz = []
            for j in range(len(y)):
                if y[j] == i:
                    scatx.append(x[j][0])
                    scaty.append(x[j][1])
                    scatz.append(x[j][2])
            if i >= len(colors):
                color = 'black'
            else:
                color = colors[i]
            ax.scatter(scatx,scaty,scatz, color=color)

        ax.legend(list(range(max(y)+1)))
        
        logger.debug("Loading Figure")
        plt.title(f'DBSCAN on {reduction}')
        plt.savefig(os.path.join(figDir + "DBSCAN-" + reduction + '-' + currentTime))
        plt.clf()
    
    # Variables
    version='2.0'

    # PCA
    logging.info('Staging PCA')
    print('Staging PCA')
    xPCA = np.load(os.path.join('reducedDims/pca/V' + version + '/pca.npy')).tolist()
    # xPCA = random.sample(xPCA, 10000)
    PCA_pred = cluster(xPCA, 'pca', 0.1, 6)
    makeFig(xPCA, PCA_pred, 'pca')
    np.save(os.path.join('results/dbscan/dbscan_pca_' + currentTime), PCA_pred)
    
    # Autoencoder
    # logging.info('Staging AE')
    # print('Staging AE')
    # xAE = np.load(os.path.join('reducedDims/autoencoder/V' + version + '/autoencoder.npy')).tolist()
    # # xAE = random.sample(xAE, 10000)
    # AE_pred = cluster(xAE, 'ae', 0.001, 1)
    # makeFig(xAE, AE_pred, 'ae')
    # np.save(os.path.join('results/dbscan/dbscan_ae_' + currentTime), AE_pred)

    # SOM
    logging.info('Staging SOM')
    print('Staging SOM')
    xSOM = np.load(os.path.join('reducedDims/som/V' + version + '/som.npy')).tolist()
    # xSOM = random.sample(xSOM, 10000)
    SOM_pred  = cluster(xSOM, 'som', 0.00001, 1)
    makeFig(xSOM, SOM_pred, 'som')
    np.save(os.path.join('results/dbscan/dbscan_som_' + currentTime), SOM_pred)

    # Performance
    # performance = performance_for_algorithm('DBSCAN', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)
    # np.save(os.path.join('results/dbscan/' + currentTime), performance)


if __name__ == "__main__":
    main()
