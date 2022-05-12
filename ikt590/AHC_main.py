from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
#from performance import performance_for_algorithm
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import logging
import random
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

    def AHC(x, k):
        ahc = AgglomerativeClustering(n_clusters=k, distance_threshold=None)
        return ahc.fit_predict(x), ahc
    

    def cluster(x, reduction, k=3, figDir='./.figs/'):
        plt.clf()
        logger.debug(f'AHC for {reduction}')
        dbscan_pred, cModel = AHC(x, k)

        # Color palette made with: https://mokole.com/palette.html
        # Settings: 16, 5%, 90%, 15000. Result: Perceived distance of 50.84, 46233 loops
        colors = ['darkslategray','maroon', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']
        
        ax = plt.axes(projection='3d')

        for i in range(k):
            scatx = []
            scaty = []
            scatz = []
            for j in range(len(dbscan_pred)):
                if dbscan_pred[j] == i:
                    scatx.append(x[j][0])
                    scaty.append(x[j][1])
                    scatz.append(x[j][2])
            
            ax.scatter(scatx,scaty,scatz, color=colors[i])
        
        ax.legend(list(range(k)))
        logger.debug("Loading Figure")
        plt.title(f'AHC on {reduction}')
        plt.savefig(os.path.join(figDir + "Hierarchical-" + reduction + '-' + currentTime))

        #return dbscan_pred, cModel
        return silhouette_score(x, dbscan_pred) # Used only for elbow method


    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        plt.show()
        plt.clf
    

    def elbow_method(x, reduction):
        max_k = 30
        scores = []
        for k in range(2,max_k):
            scores.append(cluster(x, reduction, k))

        plt.plot(scores)
        plt.show()
        plt.clf()


    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/V2.0/pca.npy').tolist()
    xPCA = random.sample(xPCA, 10000*5)
    PCA_pred = elbow_method(xPCA, 'pca') # Used only for elbow method
    #PCA_pred, PCA_model = cluster(xPCA, 'pca', k=6) # 6 clusters
    # plot_dendrogram(PCA_model, truncate_mode='level', p=3)

    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/V2.0/autoencoder.npy').tolist()
    xAE = random.sample(xAE, 10000*5)
    AE_pred = elbow_method(xAE, 'autoencoder') # Used only for elbow method
    #AE_pred, AE_model = cluster(xAE, 'autoencoder', k=4) # 4 clusters
    # plot_dendrogram(AE_model, truncate_mode='level', p=3)

    #SOM
    logging.info('Staging SOM')
    xSOM = np.load('reducedDims/som/V2.0/som.npy').tolist()
    xSOM = random.sample(xSOM, 10000*5)
    SOM_pred  = elbow_method(xSOM, 'SOM') # Used only for elbow method
    #SOM_pred, SOM_model  = cluster(xSOM, 'SOM', k=5) # 5 or 8 clusters
    # plot_dendrogram(SOM_model, truncate_mode='level', p=3)
    
    #performance = performance_for_algorithm('AHC', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)

if __name__ == "__main__":
    main()