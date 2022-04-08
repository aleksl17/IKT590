import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os

from sklearn.cluster import DBSCAN

from helpers.performance import performance_for_algorithm

np.set_printoptions(threshold=np.inf)


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

        return dbscan.fit_predict(x) # Returns -1 if sample is "noisy". See documentation.
    
    def cluster(x, reduction, eps, min_samples):
        # logger.debug(f'Kmeans for {reduction}')
        dbscan_pred = dbscan(x, eps, min_samples)

        return dbscan_pred
    
    def makeFig(x, y, reduction, figDir='./.figs/'):
        print("makeFig")
        # Color palette made with: https://mokole.com/palette.html
        # Settings: 16, 5%, 90%, 15000. Result: Perceived distance of 50.84, 46233 loops
        colors = ['darkslategray','maroon', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']

        # print("Before axes")
        # ax = plt.axes(projection='3d')
        # print(f"Length of x: {len(x)}. Length of y: {len(y)}")
        # for point, c in zip(x, y):
        #     print(c)
        #     ax.scatter3D(point[0], point[1], point[2], c=colors[c])
        # print("After axes")

        print("Before plot")
        ax = plt.axes(projection='3d')
        print("After plot")
        print("Before double for")
        for i in range(max(y)+1):
            scatx = xAE[:, 0]
            scaty = xAE[:, 1]
            scatz = xAE[:, 2]
            if i >= len(colors):
                color = 'black'
            else:
                color = colors[i]
            ax.scatter(scatx,scaty,scatz, color=color)
        print("After double for")

        ax.legend(list(range(max(y)+1)))
        
        logger.debug("Loading Figure")
        plt.title(f'DBSCAN on {reduction}')
        plt.savefig(os.path.join(figDir + "DBSCAN-" + reduction + '-' + currentTime))
        plt.clf()
    
    # Variables
    version='2.0'

    # PCA
    # logging.info('Staging PCA')
    # print('Staging PCA')
    xPCA = np.load(os.path.join('reducedDims/pca/V' + version + '/pca.npy'))
    ax = plt.axes(projection='3d')
    scatx = []
    scaty = []
    scatz = []
    print(len(xPCA))
    print(xPCA.shape)
    scatx = xPCA[:, 0]
    scaty = xPCA[:, 1]
    scatz = xPCA[:, 2]
    ax.scatter(scatx,scaty,scatz)
    plt.savefig(os.path.join('./.figs/' + "DBSCAN-" + 'PCA' + '-' + currentTime))
    plt.clf()
    # # xPCA = random.sample(xPCA, 10000)
    colors = ['dimgray','saddlebrown0','darkgreen','olive','darkslateblue','darkcyan','darkblue','darkseagreen','darkmagenta','maroon','red','darkorange','gold','lime','blueviolet','springgreen','crimson','aqua','deepskyblue','blue','greenyellow','lightsteelblue','fuchsia','dodgerblue','khaki','salmon','lightgreen','deeppink','mediumslateblue','violet','lightpink']
    PCA_pred = cluster(xPCA, 'pca', 0.1, 1)
    for point, c in zip(xPCA, PCA_pred):
        if c >= len(colors):
                color = 'black'
        else:
            color = colors[c]
        ax.scatter3D(point[0], point[1], point[2], c=color)
    plt.savefig(os.path.join('./.figs/' + "DBSCAN-" + 'PCA' + '-asdf' + currentTime))
    # print(max(PCA_pred))
    # makeFig(xPCA, PCA_pred, 'pca')
    # np.save(os.path.join('results/dbscan/dbscan_pca_' + currentTime), PCA_pred)
    
    # Autoencoder
    # logging.info('Staging AE')
    # print('Staging AE')
    xAE = np.load(os.path.join('reducedDims/autoencoder/V' + version + '/autoencoder.npy'))
    ax = plt.axes(projection='3d')
    scatx = []
    scaty = []
    scatz = []
    print(len(xAE))
    print(xAE.shape)
    scatx = xAE[:, 0]
    scaty = xAE[:, 1]
    scatz = xAE[:, 2]
    ax.scatter(scatx,scaty,scatz)
    plt.savefig(os.path.join('./.figs/' + "DBSCAN-" + 'AE' + '-' + currentTime))
    plt.clf()
    # # xAE = random.sample(xAE, 10000)
    # AE_pred = cluster(xAE, 'ae', 0.001, 1)
    # makeFig(xAE, AE_pred, 'ae')
    # np.save(os.path.join('results/dbscan/dbscan_ae_' + currentTime), AE_pred)

    # iList = []
    # # SOM
    # logging.info('Staging SOM')
    # print('Staging SOM')
    # for m in range(1, 100):
    #     for e in range(1, 20):
    #         xSOM = np.load(os.path.join('reducedDims/som/V' + version + '/som.npy')).tolist()
    #         SOM_pred  = cluster(xSOM, 'som', m/1000, e)
    #         # makeFig(xSOM, SOM_pred, 'som')
    #         # np.save(os.path.join('results/dbscan/dbscan_som_' + currentTime), SOM_pred)
    #         print(max(SOM_pred))
    #         print(min(SOM_pred))
    #         # print(len(SOM_pred))
    #         # logger.debug(np.asarray(SOM_pred).shape)
    #         # print(SOM_pred[len(SOM_pred/2)])
    #         if max(SOM_pred) > 0:
    #             if min(SOM_pred) > 0:
    #                 iList.append([max(SOM_pred), m/1000, e])
    
    # # print(iList)
    # iList = np.asarray(iList)
    # logger.info('iList:')
    # logger.info(iList)

    # # SOM
    # logging.info('Staging SOM')
    # print('Staging SOM')
    xSOM = np.load(os.path.join('reducedDims/som/V' + version + '/som.npy'))
    # logger.debug(xSOM)
    ax = plt.axes(projection='3d')
    scatx = []
    scaty = []
    scatz = []
    print(len(xSOM))
    print(xSOM.shape)
    scatx = xSOM[:, 0]
    scaty = xSOM[:, 1]
    scatz = xSOM[:, 2]
    ax.scatter(scatx,scaty,scatz)
    plt.savefig(os.path.join('./.figs/' + "DBSCAN-" + 'SOM' + '-' + currentTime))
    plt.clf()
    # SOM_pred  = cluster(xSOM, 'som', 0.0007, 7)
    # logger.debug(np.asarray(SOM_pred))
    # makeFig(xSOM, SOM_pred, 'som')
    # np.save(os.path.join('results/dbscan/dbscan_som_' + currentTime), SOM_pred)

    # Performance
    # performance = performance_for_algorithm('DBSCAN', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)
    # np.save(os.path.join('results/dbscan/' + currentTime), performance)


if __name__ == "__main__":
    main()
