from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import time

# NOTE
# Kan denne flyttes til "helpers" folderen?

def get_all_performance_for_method(x, labels):
    sil = silhouette_score(x, labels)
    dbs = davies_bouldin_score(x, labels)
    chs = calinski_harabasz_score(x, labels)

    return [sil, dbs, chs]


def performance_for_algorithm(name, PCAx, PCAy, AEx, AEy, SOMx, SOMy, saveFig = True):
    currentTime = str(int(time.time()))

    if max(PCAy) == 0:
        PCA = 0
    else:
        PCA = get_all_performance_for_method(PCAx, PCAy)

    if max(AEy) == 0:
        AE = 0
    else: 
        AE = get_all_performance_for_method(AEx, AEy)

    if max(SOMy) == 0:
        SOM = 0
    else:
        SOM = get_all_performance_for_method(SOMx, SOMy)

    scores = ['Silhouette', 'Davies Bouldin', 'Calinksi Harabasz']

    if saveFig:
        # fig = plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=3)
        for i in range(len(scores)):
            ax[i].bar(['PCA', 'Autoencoder', 'SOM'], [PCA[i], AE[i], SOM[i]], color=['gold','aquamarine','indigo'])
            ax[i].tick_params(labelrotation=20)
            ax[i].set_title(scores[i])
        
        plt.savefig(f'.figs/{name}_performance_scores_{currentTime}')
        #plt.show()
    
    return [PCA, AE, SOM]

# performance_for_algorithm('test',[[1,2],[3,4],[2,2]],[0,1,0],[[1,2],[3,4],[2,2]],[0,1,0],[[1,2],[3,4],[2,2]],[0,1,0])