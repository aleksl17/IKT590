import logging
import requests
import pandas
import numpy
import json
import time
import os

# TODO:


# def import_data(dataDirectory='Data'):
def import_data(saveLocal=True):
    """Imports data from given data directory"""
    logging.getLogger(__name__)

    # Variables
    getTime = str(int(time.time()))
    if not saveLocal:
        returnData = numpy.array([])
        

    # start="2018-01-11T08:38:13.526Z"
    # stop="2025-03-12T08:38:13.526Z"
    # apiId="0a2dd1d5-3e32-45e3-95ed-68cb2b44817e"
    # # url=f"https://signalapi.bitme.sh/api/sevents/entity/{apiId}?from={start}&to={stop}"
    # url=f"https://signalapi.bitme.sh/api/sevents/entity/{apiId}"
    # r = requests.get(url)
    # logging.info(r.status_code)
    # logging.info(r.headers['content-type'])
    # logging.info(r.encoding)
    # # logging.info(r.text)
    # # logging.info(r.json())
    # data = r.json()
    # DFData = pandas.json_normalize(data)
    # # print(DFData)

    # Read signals.json
    with open('./testsignals.json') as file:
        signalsData = json.load(file)
    
    # Export or save signal data
    for signal in signalsData:
        x = ""
        url=f'https://signalapi.bmesh.io/api/sevents/entity/{signal}'
        r = requests.get(url)
        logging.info(f"Signal: {signal} Response: {r}")
        rawData = r.json()
        for data in rawData:
            x += f"{data['timestamp']}, {data['value']}\n"

        if saveLocal:
            if not os.path.exists('./.localData'):
                os.makedirs('./.localData')
            with open(f'./.localData/{signal}_{getTime}.csv', 'w') as file:
                file.write(x)
        else:
            returnData = numpy.append(returnData, data)
