import logging
import pandas
import numpy
import os

import data_interpolate


def manipulate_data(csvDirectory='./.localData', interval=1440):
    """Manipulates data"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    # Variables
    dataList = []
    new_y = []
    x = numpy.array([])
    y = numpy.array([])

    # Creates list of DataFrames from data from CSV files in csvDirectory
    for file in os.listdir(csvDirectory):
        csvData = pandas.read_csv(os.path.join(csvDirectory, file))
        dataList.append(csvData)

    for listData in dataList:
        interpolatedData = data_interpolate.interpolation(listData)
        new_y.append(interpolatedData)
    
    print(new_y[0])
    logger.debug(type(new_y[0]))
    logger.debug(len(new_y[0]))
    logger.debug(new_y[0])
