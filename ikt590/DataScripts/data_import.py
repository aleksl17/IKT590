import requests
import logging
import pandas
import numpy
import json
import os
import io


def import_data(signalFrom="", signalTo="", signalsFile='./testsignals.json', saveDirectory="./.localData", saveLocal=True, override=False):
    """Fetches data via API and saves locally or returns list of strings"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    if saveLocal:
        # Create appropriate directories
        try:
            os.listdir(saveDirectory)
        except FileNotFoundError:
            logger.info(f"Directory not found: \"{saveDirectory}\"")
            os.makedirs(saveDirectory)
            logger.info(f"Directory created: \"{saveDirectory}\"")
        # Override
        if override:
            for file in os.listdir(saveDirectory):
                os.remove(os.path.join(saveDirectory, file))
                logger.debug(f"Removed all files in directory: \"{saveDirectory}\"")
        elif not override and os.listdir(saveDirectory):
            logger.info(f"Files already generated. Pass parameter \"override=True\" to override.")
            return
    elif not saveLocal:
        # Variables
        returnData = numpy.array([])

    # Read signals.json
    try:
        with open(signalsFile) as file:
            if signalsFile.endswith('.json'):
                signalsData = json.load(file)
            else:
                logger.error(f"File needs to end with \".json\": \"{signalsFile}")
                return
    except FileNotFoundError:
        logger.error(f"Invalid or non-existent signals file: \"{signalsFile}\"")
        return

    # Save or return signal data
    for signal in signalsData:
        csvData = "timestamp,value\n"
        url=f'https://signalapi.bmesh.io/api/sevents/entity/{signal}'
        # "from" and "to" parameter URL constructing logic 
        if signalFrom or signalTo:
            url += "?"
            if signalFrom:
                url += f"from={signalFrom}"
            if signalFrom and signalTo:
                url += "&"
            if signalTo:
                url += f"to={signalTo}"
        logger.debug(f"URL: {url}")
        # Get data from constructed URL above and parse
        r = requests.get(url)
        rawData = r.json()
        # Create csv data string from parsed JSON
        for data in rawData:
            # Only appends temperature
            if data['type'] == "VEvent" and not data['sourceId'].split("/")[-1] == "hu":
                csvData += f"{data['timestamp'][:-1]},{data['value']}\n"
        # Save data locally or append to numpy array
        if saveLocal:
            with open(f'{saveDirectory}/{signal}.csv', 'w') as file:
                file.write(csvData)
                logger.info(f"Wrote file: \"{signal}.csv\"")
        else:
            returnDataDF = pandas.read_csv(io.StringIO(csvData), sep=',')
            returnData = numpy.append(returnData, returnDataDF)

    if not saveLocal:
        return returnData
