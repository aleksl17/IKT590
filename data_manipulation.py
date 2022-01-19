import logging
import pandas
import numpy
import os

# TODO:
# Get JSON data without backslash escape characters. LOL


def import_local_data(dataDirectory='Data'):
    """Imports data from given data directory"""

    if not os.path.exists('Data'):
        print("Directory not found")
    
    # Variables
    dataArray = numpy.array([])
    logging.debug('test')

    # Read local data
    for filename in os.listdir(dataDirectory):
        dataPath = os.path.join(dataDirectory, filename)
        # print(dataPath)
        tmpData = pandas.read_json(dataPath)
        print(tmpData)
        dataArray = numpy.append(dataArray, tmpData)

    # print(dataArray)


def clean_data(interval=1440):
    print("clean data")
