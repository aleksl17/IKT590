import logging
import pandas
import time
import json
import os

import data_interpolate

# TODO:
# Optimization: Use Numpy arrays instead of Python lists


def create_dataset(inputDirectory='./.localData/', outputDirectory='./.dataset/', sample_size=40):
    """Manipulates data"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    # Variables
    currentTime = str(int(time.time()))
    datasetList = []
    metadata = []
    samples = []

    # Creates list of DataFrames from data from CSV files input directory
    for file in os.listdir(inputDirectory):
        # Read files and interpolate
        csvData = pandas.read_csv(os.path.join(inputDirectory, file))
        data = data_interpolate.interpolation(csvData)
        data = data.tolist()

        logger.debug(type(data))
        # logger.debug(data)
        logger.debug(len(data))
        # logger.debug(numpy.shape(data))
        
        # Split data into samples with overlap and create respective metadata tabel
        for set in range(len(data)-sample_size):
            samples.append(data[set:set+sample_size])
            metadata.append([file.split('.csv')[0], csvData['timestamp'][set], data[set]])

        # Create list of datasets
        datasetList.append([metadata, samples])
        
        # Create ouput directory and write dataset to JSON file in said directory
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)
        with open(os.path.join(outputDirectory+'dataset-'+currentTime+'.json'), 'w') as filehandle:
            filehandle.write(json.dumps(datasetList))


def read_dataset(datasetFile='./.dataset/datasetV1.0.json', returnType='list'):
    """Reads data from file and returns it"""

    # Variables
    datasetList = []

    # Opens dataset file and reads content to a python list
    with open(datasetFile, 'r') as filehandle:
        datasetList = json.loads(filehandle.read())

    # Return dataset in preferred format
    if returnType=='list':
        return datasetList
    else:
        print(f"Invalid \"returnType\": {returnType}")
        return
