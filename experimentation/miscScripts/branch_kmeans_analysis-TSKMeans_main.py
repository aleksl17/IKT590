import logging
import random
import time
import os
from turtle import color
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
# from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import helpers.data_manipulation as data_manipulation
import numpy as np

import helpers.data_interpolate as interpolate

# TODO:
# Remove unused imports
# Remove unused code
# Team Review


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    

    def kMeans(data, clusters, metric='softdtw', plot=False, saveDirectory="./.figs"):
        sz = data.shape[1]
        data = data[:100]
        model = TimeSeriesKMeans(n_clusters=clusters,
                                n_init=2,               # Rememba to change (was 2)
                                metric=metric,
                                verbose=True,
                                max_iter_barycenter=100,
                                random_state=0)
        y_pred = model.fit_predict(data)
        if(plot): 
            # plt.figure()
            # for yi in range(clusters):
            #     plt.subplot(clusters, yi + 1)
            #     for xx in data[y_pred == yi]:
            #         plt.plot(xx.ravel(), "k-", alpha=.2)
            #     plt.plot(model.cluster_centers_[yi].ravel(), "r-")
            #     # plt.xlim(0, sz)
            #     # plt.ylim(0, 100)
            #     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
            #             transform=plt.gca().transAxes)
            #     if yi == 1:
            #         plt.title(metric)
            # plt.tight_layout()
            # # plt.show()
            # if not os.path.exists(saveDirectory):
            #     os.makedirs(saveDirectory)
            # plt.savefig(f'{saveDirectory}/K{clusters}_{currentTime}')
            # #print clusters centers
            # # for c in model.cluster_centers_:
            # #     plt.plot(c)

            plt.clf()
            fig, axs = plt.subplots(k)
            fig.suptitle('clusters')

            for i, ax in enumerate(axs):
                ax.set_title("Cluster: " + str(i))

            for c, s in zip(y_pred, data):
                axs[c].plot(s, color='black', alpha=.2  )

            for c in range(k):
                axs[c].plot(model.cluster_centers_[c], color='r')
            
            fig.tight_layout()
            # plt.savefig(os.path.join('./.figs/' + "KMeans-AE-real"  + currentTime))
            plt.show()

            # plt.show()

        #silhouette score
        return silhouette_score(data, y_pred, metric=metric), model.cluster_centers_

    meta, dataset = data_manipulation.read_dataset(datasetFile='./.dataset/dataset.json')

    dataset = random.sample(dataset, 10000)

    sil_scores = []
    cluster_centers = []
    x_train = np.asarray(dataset)

    for k in range(2,30):
        sil_score, cluster_center = kMeans(x_train, k, metric='dtw', plot=False)
        sil_scores.append(sil_score)
        cluster_centers.append(cluster_center)
        # plt.clf()
        # for c in cluster_center:
        #     plt.plot(c)
        # plt.savefig(f'./.figs/clusters_K_{k}_{currentTime}')


    xList = range(2,30)
    plt.clf()   
    plt.plot(xList, sil_scores)
    plt.savefig(f'./.figs/sil_score_{currentTime}')
    # plt.show()  

if __name__ == "__main__":
    main()
