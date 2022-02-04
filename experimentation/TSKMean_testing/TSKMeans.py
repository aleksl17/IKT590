from operator import mod
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from data import get_data

#must have nupmy v=1.21 
seed = 0
np.random.seed(seed)

#params
X_train = np.asarray(get_data())
# X_train = X_train[:50]
sz = X_train.shape[1]

def kMeans(data, clusters, metric='dtw', plot=False):
    model = TimeSeriesKMeans(n_clusters=clusters,
                            n_init=2,
                            metric=metric,
                            verbose=True,
                            max_iter_barycenter=10,
                            random_state=seed)
    y_pred = model.fit_predict(X_train)

    if(plot): 
        plt.figure()
        for yi in range(clusters):
            plt.subplot(1, clusters, yi + 1)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(model.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(0, 100)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.title(metric)

        plt.tight_layout()
        plt.show()

        #print clusters centers
        for c in model.cluster_centers_:
            plt.plot(c)

        plt.show()

    #silhouette score
    return silhouette_score(X_train, y_pred, metric=metric), model.cluster_centers_

sil_scores = []
cluster_centers = []

for k in range(2,10):
    sil_score, cluster_center = kMeans(X_train, k, metric='soft dtw')
    sil_scores.append(sil_score)
    cluster_centers.append(cluster_center)

plt.plot(sil_scores)
plt.show()  