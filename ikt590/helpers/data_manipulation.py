import logging
import pandas
import time
import json
import os

import helpers.data_interpolate as data_interpolate
import helpers.data_normalize as data_normalize

# TODO:
# Optimization: Use Numpy arrays instead of Python lists
# NOTE
# Reference and dataset might not align due to time-shift from interpolation.-


def create_dataset(inputDirectory='./.tmpData/', outputDirectory='./.tmpData/', sample_size=40):
    """Manipulates data"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)

    # Initialize directories
    if not os.path.exists(inputDirectory):
        os.makedirs(inputDirectory)
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    
    # Variables
    currentTime = str(int(time.time()))
    datasetList = []
    datasetMetadata = []
    samples = []
    reference = []
    referenceMetadata = []

    # Creates list of DataFrames from data from CSV files input directory
    for file in os.listdir(inputDirectory):
        if file.endswith('.csv'):
            # Read files
            csvData = pandas.read_csv(os.path.join(inputDirectory, file))
            valueData = csvData['value'].tolist()

            # Create reference "samples"
            for i in range(len(valueData)-sample_size):
                reference.append(valueData[i:i+sample_size])
                referenceMetadata.append([file.split('.csv')[0], csvData['timestamp'][i], valueData[i]])

            # Interpolate data and convert to python list
            intData = data_interpolate.interpolation(csvData)
            intData = intData.tolist()
            
            # Normalize data
            normData = data_normalize.normalize(intData)

            # Split data into samples with overlap and create respective metadata tabel
            for set in range(len(normData)-sample_size):
                samples.append(normData[set:set+sample_size])
                datasetMetadata.append([file.split('.csv')[0], csvData['timestamp'][set], normData[set]])

            # Create list of datasets
            # datasetList.append([metadata, samples])
    
    # Create list of dataset metadata and samples
    datasetList = [datasetMetadata, samples]
    # Create list of reference metadata and data
    referenceList = [referenceMetadata, reference]

    # logger.debug(f"datasetList first:\n{datasetList[:1]}")
    # logger.debug(f"referenceList first:\n{referenceList[:1]}")

    # Create ouput directory
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    # Write dataset
    with open(os.path.join(outputDirectory+'dataset-'+currentTime+'.json'), 'w') as filehandle:
        filehandle.write(json.dumps(datasetList))
    # Write reference
    with open(os.path.join(outputDirectory+'reference-'+currentTime+'.json'), 'w') as filehandle:
        filehandle.write(json.dumps(referenceList))


def read_dataset(datasetFile='datasets/dataset.json', returnType='list'):
    """Reads data from file and returns it"""

    # Variables
    list = []

    # Opens data file and reads content to a python list
    with open(datasetFile, 'r') as filehandle:
        list = json.loads(filehandle.read())

    # Return data in preferred format
    if returnType=='list':
        metadata = list[0]
        x = list[1]
        return metadata, x
    else:
        print(f"Invalid \"returnType\": {returnType}")
        return
