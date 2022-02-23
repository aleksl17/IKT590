import logging
import random
import pandas
import time
import math
import os
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import numpy as np
import sys

import scripts.data_interpolate as interpolate

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
    
    # Variables
    dataset = []
    sample_size = 24 * 4 # since each sample length is 15 minutes

    def kMeans(data, clusters, metric='dtw', plot=False, saveDirectory="./.figs"):
        sz = data.shape[1]
        data = data[:100]
        model = TimeSeriesKMeans(n_clusters=clusters,
                                n_init=1,               # Rememba to change (was 2)
                                metric=metric,
                                verbose=True,
                                max_iter_barycenter=10,
                                random_state=0)
        y_pred = model.fit_predict(data)
        if(plot): 
            plt.figure()
            for yi in range(clusters):
                plt.subplot(1, clusters, yi + 1)
                for xx in data[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(model.cluster_centers_[yi].ravel(), "r-")
                # plt.xlim(0, sz)
                # plt.ylim(0, 100)
                plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                        transform=plt.gca().transAxes)
                if yi == 1:
                    plt.title(metric)
            plt.tight_layout()
            # plt.show()
            if not os.path.exists(saveDirectory):
                os.makedirs(saveDirectory)
            plt.savefig(f'{saveDirectory}/K{clusters}_{currentTime}')
            #print clusters centers
            # for c in model.cluster_centers_:
            #     plt.plot(c)

            # plt.show()
        #silhouette score
        return silhouette_score(data, y_pred, metric=metric), model.cluster_centers_


    for filename in os.listdir('./.tmpData'):
        csvData = pandas.read_csv(f"./.tmpData/{filename}")
        # print(csvData)
        data = interpolate.interpolation(csvData)
        # for i in range(math.floor(len(data)/sample_size)):
        #     dataset.append(data[i*sample_size:(i+1)*sample_size])
        for i in range(len(data)-sample_size):
            dataset.append(data[i:i + sample_size])

    dataset = random.sample(dataset, 10000)

    sil_scores = []
    cluster_centers = []
    x_train = np.asarray(dataset)

    for k in range(2,10):
        sil_score, cluster_center = kMeans(x_train, k, metric='softdtw', plot=True)
        sil_scores.append(sil_score)
        cluster_centers.append(cluster_center)
        plt.clf()
        for c in cluster_center:
            plt.plot(c)
        plt.savefig(f'./.figs/clusters_K_{k}_{currentTime}')

    plt.clf()   
    plt.plot(sil_scores)
    plt.savefig(f'./.figs/sil_score_{currentTime}')
    # plt.show()  

if __name__ == "__main__":
    main()
