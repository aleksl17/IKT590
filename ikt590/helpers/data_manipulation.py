import logging
import pandas
import time
import json
import os

# import data_interpolate
# import data_interpolate
import scripts.data_interpolate as data_interpolate
import scripts.data_normalize as data_normalize

# TODO:
# Optimization: Use Numpy arrays instead of Python lists


def create_dataset(inputDirectory='./.tmpData/', outputDirectory='./.tmpData/', sample_size=40):
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
        # Read files
        csvData = pandas.read_csv(os.path.join(inputDirectory, file))

        # Interpolate data and convert to python list
        intData = data_interpolate.interpolation(csvData)
        intData = intData.tolist()
        
        # Normalize data
        data = data_normalize.normalize(intData)

        # Split data into samples with overlap and create respective metadata tabel
        for set in range(len(data)-sample_size):
            samples.append(data[set:set+sample_size])
            metadata.append([file.split('.csv')[0], csvData['timestamp'][set], data[set]])

        # Create list of datasets
        # datasetList.append([metadata, samples])
    
    # Create list of metadata and samples
    datasetList = [metadata, samples]

    # Create ouput directory and write dataset to JSON file in said directory
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    with open(os.path.join(outputDirectory+'dataset-'+currentTime+'.json'), 'w') as filehandle:
        filehandle.write(json.dumps(datasetList))


def read_dataset(datasetFile='./.dataset/dataset-1644394453.json', returnType='list'):
    """Reads data from file and returns it"""

    # Variables
    datasetList = []

    # Opens dataset file and reads content to a python list
    with open(datasetFile, 'r') as filehandle:
        datasetList = json.loads(filehandle.read())

    # Return dataset in preferred format
    if returnType=='list':
        metadata = datasetList[0]
        x = datasetList[1]
        return metadata, x
    else:
        print(f"Invalid \"returnType\": {returnType}")
        return
