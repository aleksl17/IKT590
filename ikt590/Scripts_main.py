from matplotlib import projections
import matplotlib.pyplot as plt
import logging
import time
import random
import os
import pandas
import numpy

import helpers.data_import as data_import
import helpers.data_manipulation as data_manipulation
import helpers.data_fft as data_fft

def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    
    logging.info('Started')
    ################ ======== Functions/code goes below here ======== ################

    def reducedDimsMakeFig(reducedDimsPath, RDType, figDir='./.figs/'):
        reducedDim = numpy.load(reducedDimsPath)
        numpy.random.shuffle(reducedDim)
        plt.clf()
        ax = plt.axes(projection='3d')
        for x, y, z in reducedDim[:1000]:
            ax.scatter(x, y, z, color=numpy.random.rand(3,))
        plt.title(f'Reduced Dimensions from {RDType}')
        plt.savefig(os.path.join(figDir + RDType + '-' + currentTime))
    
    # Download data via API if .localData folder is empty
    #import_data.import_data(signalFrom="2021-01-01T01:00:00.000Z", signalTo="2022-01-01T01:00:00.000Z")
    # idd = data_import.import_data()
    # print(idd)

    # Create dataset from local data files
    # dmc = data_manipulation.create_dataset()
    # print(dmc)

    # fft = data_fft.fourier_transform(pandas.read_csv("./signals/2f96cc4c-5ad4-6b3d-234c-e1073e85bd95.csv"))
    # print(fft)

    # Reduced dims figures
    # rdmf_ae = reducedDimsMakeFig('reducedDims/autoencoder/V2.0/autoencoder.npy', 'AE')
    # rdmf_pca = reducedDimsMakeFig('reducedDims/pca/V2.0/pca.npy', 'PCA')
    # rdmf_som = reducedDimsMakeFig('reducedDims/som/V2.0/som.npy', 'SOM')

    # Read dataset from local dataset files
    # dmr = data_manipulation.read_dataset()
    # logging.debug(dmr)
    # logging.debug(len(dmr))
    # logging.debug(len(dmr[0]))
    # logging.debug(len(dmr[0][0]))
    # logging.debug(len(dmr[0][0][0]))
    # logging.debug(len(dmr[0][0][0][0]))
    # logging.debug(len(dmr[0][0][0][1]))
    # logging.debug(len(dmr[0][0][0][0][0]))
    # logging.debug(len(dmr[0][0][0][1][0]))
    # logging.debug(len(dmr[0][0][0][0][0][0]))
    # logging.debug(len(dmr[0][0][0][0][0][0][0]))
    # logging.debug(dmr[0][1])
    
    ################ ======== Functions/code goes above here ======== ################
    logging.info('Finished')


if __name__ == "__main__":
    main()
