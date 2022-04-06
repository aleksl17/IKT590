from sklearn.cluster import AgglomerativeClustering
import helpers.data_manipulation as data_manipulation
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

    def AHC(x, k):
        # ahc = AgglomerativeClustering(n_clusters=n_clusters, compute_full_tree=True)
        ahc = AgglomerativeClustering(n_clusters=k, distance_threshold=None)
        return ahc.fit_predict(x), ahc
    
    def cluster(x, reduction, k, figDir='./.figs/'):
        logger.debug(f'AHC for {reduction}')
        dbscan_pred, cModel = AHC(x, k)

        # Color palette made with: https://mokole.com/palette.html
        # Settings: 16, 5%, 90%, 15000. Result: Perceived distance of 50.84, 46233 loops
        colors = ['darkslategray','maroon', 'darkgreen', 'darkkhaki', 'darkblue', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']
        
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

    meta, dataset = data_manipulation.read_dataset(datasetFile='datasets/V2.0/dataset.json')
    

    #PCA
    logging.info('Staging PCA')
    xPCA = np.load('reducedDims/pca/V2.0/pca.npy').tolist()
    xPCA = random.sample(xPCA, 10000)
    tmpData, xPCA = zip(*random.sample(list(zip(dataset, xPCA)),100))
    PCA_pred, PCA_model = cluster(xPCA, 'pca', k=6)
    # plot_dendrogram(PCA_model, truncate_mode='level', p=3)

    plt.clf()
    fig, axs = plt.subplots(6)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi,hi*2)
    fig.suptitle('clusters')

    for i, ax in enumerate(axs):
        ax.set_title("Cluster: " + str(i))

    for c, s in zip(PCA_pred, tmpData):
        axs[c].plot(s, color='black', alpha=.2)
    
    fig.tight_layout()
    plt.savefig(os.path.join('./.figs/' + "AHC-PCA-real"  + currentTime))


    #Autoencoder
    logging.info('Staging AE')
    xAE = np.load('reducedDims/autoencoder/V2.0/autoencoder.npy').tolist()
    xAE = random.sample(xAE, 10000)
    tmpData, xAE = zip(*random.sample(list(zip(dataset, xAE)),100))
    AE_pred, AE_model = cluster(xAE, 'autoencoder', k=4)
    # plot_dendrogram(AE_model, truncate_mode='level', p=3)

    plt.clf()
    fig, axs = plt.subplots(4)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi,hi*2)
    fig.suptitle('clusters')

    for i, ax in enumerate(axs):
        ax.set_title("Cluster: " + str(i))

    for c, s in zip(AE_pred, tmpData):
        axs[c].plot(s, color='black', alpha=.2)
    
    fig.tight_layout()
    plt.savefig(os.path.join('./.figs/' + "AHC-AE-real"  + currentTime))


    #SOM
    logging.info('Staging SOM')
    xSOM = np.load('reducedDims/som/V2.0/som.npy').tolist()
    xSOM = random.sample(xSOM, 10000)
    tmpData, xSOM = zip(*random.sample(list(zip(dataset, xSOM)),100))
    SOM_pred, SOM_model  = cluster(xSOM, 'SOM', k=5) # 5 or 8
    # plot_dendrogram(SOM_model, truncate_mode='level', p=3)

    plt.clf()
    fig, axs = plt.subplots(5)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi,hi*2)
    fig.suptitle('clusters')

    for i, ax in enumerate(axs):
        ax.set_title("Cluster: " + str(i))

    for c, s in zip(SOM_pred, tmpData):
        axs[c].plot(s, color='black', alpha=.2)
    
    fig.tight_layout()
    plt.savefig(os.path.join('./.figs/' + "AHC-SOM-real"  + currentTime))

    # Performance
    performance = performance_for_algorithm('AHC', xPCA, PCA_pred, xAE, AE_pred, xSOM, SOM_pred)

if __name__ == "__main__":
    main()
