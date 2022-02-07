import logging
import pandas
import numpy
import math
import os

import data_interpolate

# TODO:
# Doesn't work if <interpolate=False>
# Function creates CSV with 2D list. Where each row is a subset of index 0 = start datetime, and 1... is values of interval
#   [
#   [2021-01-01:00:11:22, 1.4, 1.5, 1.6, ... ],
#   [2022-02-02:12:34:45, 9.9, 5.4, 7.1, ... ]
#   ]
# Expected Shape: [<floor of length of data times sample_size>, <sample_size>]


def manipulate_data(csvDirectory='./.localData', sample_size=40, interpolate=True):
    """Manipulates data"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    # Variables
    datasetList = []
    dataset = numpy.empty((0))
    # x = numpy.array([])
    # y = numpy.array([])

    # Creates list of DataFrames from data from CSV files in csvDirectory
    for file in os.listdir(csvDirectory):
        # Read files and interpolate if "interpolate==True"
        csvData = pandas.read_csv(os.path.join(csvDirectory, file))
        if interpolate:
            csvData = data_interpolate.interpolation(csvData)
        
        # Split csvData into samples (Now without loops!)
        splitAmount = math.floor(len(csvData)/sample_size)
        trimIndex = splitAmount*sample_size
        csvData = csvData[:trimIndex]
        dataset = numpy.split(csvData, splitAmount)

        # Append dataset containing samples to list
        datasetList.append(dataset)
    
    return datasetList
