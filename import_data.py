import logging
import requests
import pandas
import numpy
import json
import time
import os


def import_data(saveLocal=True):
    """Imports data from given data directory"""
    logging.getLogger(__name__)

    # Variables
    currentTime = str(int(time.time()))
    if not saveLocal:
        returnData = numpy.array([])

    # Read signals.json
    with open('./testsignals.json') as file:
        signalsData = json.load(file)
    
    # Save or return signal data
    for signal in signalsData:
        csvData = ""
        url=f'https://signalapi.bmesh.io/api/sevents/entity/{signal}'
        r = requests.get(url)
        rawData = r.json()
        for data in rawData:
            csvData += f"{data['timestamp']}, {data['value']}\n"

        # Save data locally or append to numpy array
        if saveLocal:
            if not os.path.exists('./.localData'):
                os.makedirs('./.localData')
            with open(f'./.localData/{signal}_{currentTime}.csv', 'w') as file:
                file.write(csvData)
        else:
            returnData = numpy.append(returnData, csvData)
    
    if not saveLocal:
        return returnData
