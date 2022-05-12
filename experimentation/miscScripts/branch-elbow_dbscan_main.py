from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
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

    def dbscan(x, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # x0 = random.sample(x, 1000)
        # dbscan.fit(x0)
        return dbscan.fit_predict(x)
    
    def cluster(x, reduction, eps, min_samples, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        dbscan_pred = dbscan(x, eps, min_samples)

        # colors = ['r','b','g','y','m','c','k']
        # colors = ['lightcoral', 'red', 'darkred', 'chocolate', 'bisque', 'darkorange', 'gold', 'yellow', 'olive', 'darkgreen', 'lime', 'aquamarine', 'teal', 'cyan', 'lightblue', 'steelblue', 'navy', 'blue', 'indigo', 'violet', 'purple', 'crimson', 'pink']
        # random.shuffle(colors)

        # ax = plt.axes(projection='3d')
        # for point, c in zip(x, dbscan_pred):
        #     ax.scatter3D(point[0], point[1], point[2], c=colors[c])
        
        # logger.debug("Loading Figure")
        # plt.title(f'DBSCAN on {reduction}')
        # plt.savefig(os.path.join(figDir + "DBSCAN-eps:" + str(int(eps*10)) + "-min:" + str(min_samples) + reduction + '-' + currentTime))
        # plt.clf()
        if max(dbscan_pred) == 0:
            return 0
        return silhouette_score(x, dbscan_pred)

    def parameters(x, reduction):
        eps_list = [0.1,0.3,0.5,0.7]
        min_samples_list = [3,4,5,6,7]

        sil_scores = []
        param_list = []

        for i in range(10):
            eps = random.choice(eps_list)

            dup = True
            while(dup):
                min_samples = random.choice(min_samples_list)
                dup = param_list.count([eps,min_samples]) > 0


                
                

            param_list.append([eps,min_samples])
            
            sil_scores.append(cluster(x, reduction, eps, min_samples))

        print(param_list)
        plt.clf()
        plt.plot(sil_scores)
        for p, s, i in zip(param_list,sil_scores, range(len(sil_scores))):
            plt.text(i,s, str(p))

        plt.ylabel("Silhouette Score")
        plt.xlabel("Parameters")
        plt.show()


    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/1644398105.npy').tolist()
    xPCA = random.sample(xPCA, 10000)
    parameters(xPCA, 'pca')
    
    # #Autoencoder
    # logging.info('Staging AE')
    # xAE = np.load('reducedDims/autoencoder/1644405844.npy').tolist()
    # xAE = random.sample(xAE, 10000)
    # parameters(xAE, 'autoencoder')

    # #SOM
    # logging.info('Staging SOM')
    # xSOM = np.load('reducedDims/som/1644406419.npy').tolist()
    # xSOM = random.sample(xSOM, 10000)
    # parameters(xSOM, 'SOM')


if __name__ == "__main__":
    main()
