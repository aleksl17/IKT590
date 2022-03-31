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
        
        kmeans.fit(x)
        return kmeans.predict(x)
    
    def cluster(x, reduction, k = 3, figDir='./.figs/'):
        logger.debug(f'Kmeans for {reduction}')
        kmeans_pred = kMeans(x,k, False)

        colors = ['blue','red', 'yellow', 'green', 'fuchsia', 'red', 'darkturquoise', 'orange', 'yellow', 'lime', 'mediumspringgreen', 'blue', 'thistle', 'fuchsia', 'dodgerblue', 'deeppink']

        plt.clf()
        ax = plt.axes(projection='3d')
        # for point, c in zip(x, kmeans_pred):
        #     ax.scatter3D(point[0], point[1], point[2], color=colors[c], label=c)

        for i in range(k):
            scatx = []
            scaty = []
            scatz = []
            for j in range(len(kmeans_pred)):
                if kmeans_pred[j] == i:
                    scatx.append(x[j][0])
                    scaty.append(x[j][1])
                    scatz.append(x[j][2])
            
           
            ax.scatter(scatx,scaty,scatz, color=colors[i])
        # scatter = ax.scatter3D(x,color=colors[kmeans_pred])
        ax.legend(list(range(k)))
        logger.debug("Loading Figure")
        plt.title(f'K-Means on {reduction}')
        # plt.savefig(os.path.join(figDir + "KMeans-" + reduction + '-' + currentTime))
        plt.show()
        return kmeans_pred
    

    # meta, dataset = data_manipulation.read_dataset(datasetFile='./.dataset/dataset.json')
    r = 2
    k = 10

    id = "1648729240"
    reduction = "pca"

    x = np.load(f"./saved_clusters/{reduction}/x_{id}.npy")
    data = np.load(f"./saved_clusters/{reduction}/data_{id}.npy")
    pred = np.load(f"./saved_clusters/{reduction}/pred_{id}.npy")

    
    tmpX = []
    tmpData = []

    for x1, d, p in zip(x,data,pred):
        if p == r:
            tmpX.append(x1)
            tmpData.append(d)
    

    preds = cluster(tmpX, f"{reduction}_recluster_", k)

    plt.clf()
    fig, axs = plt.subplots(k)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi,hi*k/2)
    fig.suptitle('clusters')

    for i, ax in enumerate(axs):
        ax.set_title("Cluster: " + str(i))

    for c, s in zip(preds, tmpData):
        axs[c].plot(s, color='black', alpha=.2)
    
    fig.tight_layout()
    # plt.savefig(os.path.join('./.figs/' + "KMeans-PCA-real"  + currentTime))
    plt.show()

    print(x)

   

if __name__ == "__main__":
    main()
