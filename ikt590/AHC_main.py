from sklearn.cluster import AgglomerativeClustering
from performance import performance_for_algorithm
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

    def AHC(x, n_clusters = 3):
        # ahc = AgglomerativeClustering(n_clusters=n_clusters, compute_full_tree=True)
        ahc = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        return ahc.fit_predict(x), ahc
    
    def cluster(x, reduction, k = 3, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        dbscan_pred, cModel = AHC(x)

        # Color palette made with: https://mokole.com/palette.html
        # Settings: 16, 5%, 90%, 15000. Result: Percieved distance of 50.84, 46233 loops
        colors = ['darkslategray','maroon2', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']
        
        ax = plt.axes(projection='3d')
        for point, c in zip(x, dbscan_pred):
            if c > len(colors):
                logger.warning("Max colors reached! Consider adding more colors.")
                c = len(colors)-1
            ax.scatter3D(point[0], point[1], point[2], c=colors[c])
        
        logger.debug("Loading Figure")
        plt.title(f'Hierarchical Clustering on {reduction}')
        plt.savefig(os.path.join(figDir + "Hierarchical-" + reduction + '-' + currentTime))
        plt.clf()

        return dbscan_pred, cModel

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
    
    
    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/1644398105.npy').tolist()
    xPCA = random.sample(xPCA, 10000)
    PCA_pred, PCA_model = cluster(xPCA, 'pca', k=3)
    # plot_dendrogram(PCA_model, truncate_mode='level', p=3)

    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/1644405844.npy').tolist()
    xAE = random.sample(xAE, 10000)
    AE_pred, AE_model = cluster(xAE, 'autoencoder', k=3)
    # plot_dendrogram(AE_model, truncate_mode='level', p=3)

    #SOM
    logging.info('Staging SOM')
    xSOM = np.load('reducedDims/som/1644406419.npy').tolist()
    xSOM = random.sample(xSOM, 10000)
    SOM_pred, SOM_model  = cluster(xSOM, 'SOM', k=3)
    # plot_dendrogram(SOM_model, truncate_mode='level', p=3)

    performance = performance_for_algorithm('AHC', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)

if __name__ == "__main__":
    main()
