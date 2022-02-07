import logging
from sqlite3 import Timestamp
import pandas
import numpy
import os

import data_interpolate

# TODO:
# Function creates CSV with 2D list. Where each row is a subset of index 0 = start datetime, and 1... is values of interval
#   [
#   [2021-01-01:00:11:22, 1.4, 1.5, 1.6, ... ],
#   [2022-02-02:12:34:45, 9.9, 5.4, 7.1, ... ]
#   ]

def manipulate_data(csvDirectory='./.localData', sample_size=40, interpolate=True):
    """Manipulates data"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    # Variables
    datasetList = numpy.array([])
    dataset = numpy.array([])
    # x = numpy.array([])
    # y = numpy.array([])

    # Creates list of DataFrames from data from CSV files in csvDirectory
    for file in os.listdir(csvDirectory):
        # csvData = pandas.read_csv(os.path.join(csvDirectory, file), parse_dates=['timestamp'])
        csvData = pandas.read_csv(os.path.join(csvDirectory, file))
        # csvData['Datetime']
        if interpolate:
            csvData = data_interpolate.interpolation(csvData)
        
        for set in range(len(csvData)-sample_size):
            print(set)
            numpy.append(dataset, csvData[set:set+sample_size])

        # How to convert Pandas DataFrame to Pandas Series. Might not be needed here, but good to know how.
        # Requires "timestamp" to be DateTime and not string.
        # csvDataSeries = pandas.Series(data=csvData['value'].values, index=csvData['timestamp'])

        # csvDataX, csvDataY = data_interpolate.interpolation(csvData)
        # print(csvDataX)
        # print(csvDataY)
        
        # csvData = csvData.groupby(pandas.Grouper(key='timestamp', freq='1D'))
        # print(csvData)
        
        numpy.append(datasetList, dataset)
    
print(manipulate_data())
    
    


    # print(dataList[0])
    # logger.debug(type(dataList[0]))
    # logger.debug(len(dataList[0]))
    # logger.debug(dataList[0])
