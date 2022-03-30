import logging
import pandas
import numpy
import time
import json
import os

from datetime import datetime, timedelta

import helpers.data_interpolate as data_interpolate
import helpers.data_normalize as data_normalize

# TODO:
# Optimization: Use Numpy arrays instead of Python lists


def create_dataset(inputDirectory='./signals/', outputDirectory='./.tmpData/', sample_size=40, step_size=20, outlierMulti=3):
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
    samplesMetadata = []
    referencesMetadata = []
    allSamples = []
    allReferences = []
    # postProcSamplesMetadata = []
    # postProcSamples = []
    # postProcReferencesMetadata = []
    # postProcReferences = []

    # Creates list of DataFrames from data from CSV files input directory
    for file in os.listdir(inputDirectory):
        if file.endswith('.csv'):
            # Read files
            csvData = pandas.read_csv(os.path.join(inputDirectory, file))
            valueData = csvData['value'].tolist()
            logger.debug(f"Read file: {file}")

            # Create reference "samples"
            refTime = datetime.strptime(csvData['timestamp'][0], '%Y-%m-%dT%H:%M:%S')
            referenceList = []
            # tmpReferenceList = []
            # tmpReferenceMetadataList = []
            for i in range(0, len(valueData)-sample_size, step_size):
                referenceList.append(valueData[i:i+sample_size])
                referencesMetadata.append([file.split('.csv')[0], str(refTime), valueData[i]])
                refTime = refTime + timedelta(minutes=15)
            # refMax = numpy.mean(referenceList)*outlierMulti
            # for i, reference in enumerate(referenceList):
            #     refMin = min(reference)
            #     if refMin < refMax:
            #         tmpReferenceList.append(reference)
            #         tmpReferenceMetadataList.append(referencesMetadata[i])
            # postProcReferencesMetadata.extend(tmpReferenceMetadataList)
            # postProcReferences.extend(tmpReferenceList)
            allReferences.extend(referenceList)
            
            # Interpolate data and convert to python list
            intData = data_interpolate.interpolation(csvData)
            intData = intData.tolist()
            # Dump per file data
            # with open(f".tmpData/{file}_{currentTime}.json", "w") as wfh:
            #     wfh.write(json.dumps(intData))
            
            # Normalize data
            normData = data_normalize.normalize(intData)

            # Split data into samples with overlap and create respective metadata tabel
            sampleList = []
            # tmpSampleList = []
            # tmpSampleMetadataList = []
            for set in range(0, len(normData)-sample_size, step_size):
                sampleList.append(normData[set:set+sample_size])
                samplesMetadata.append([file.split('.csv')[0], csvData['timestamp'][set], normData[set]])
            # samMax = numpy.mean(sampleList)*outlierMulti
            # for i, sample in enumerate(sampleList):
            #     samMin = min(sample)
            #     if samMin < samMax:
            #         tmpSampleList.append(sample)
            #         tmpSampleMetadataList.append(samplesMetadata[i])
            # postProcSamplesMetadata.extend(tmpSampleMetadataList)
            # postProcSamples.extend(tmpSampleList)
            allSamples.extend(sampleList)

    # Debug checks
    # logger.debug(f"referencesMaxList: {referencesMaxList}")
    # logger.debug(f"perFileSampleLengthList: {perFileSampleLengthList}")
    # logger.debug(f"referencesMetadata length: {len(referencesMetadata)}")
    # logger.debug(f"references length: {len(references)}")
    # logger.debug(f"samplesMetadata length: {len(samplesMetadata)}")
    # logger.debug(f"samples length: {len(samples)}")

    # logger.debug(f"PostProcSamplesMetadata first 5:\n{postProcSamplesMetadata[:5]}")
    # logger.debug(f"PostProcSamplesMetadata last 5:\n{postProcSamplesMetadata[-5:]}")
    # logger.debug(f"PostProcSamplesMetadata length: {len(postProcSamplesMetadata)}")
    # logger.debug(f"PostProcSamplesMetadata type: {type(postProcSamplesMetadata)}")
    # logger.debug(f"PostProcSamples first 5:\n{postProcSamples[:5]}")
    # logger.debug(f"PostProcSamples last 5:\n{postProcSamples[-5:]}")
    # logger.debug(f"PostProcSamples length: {len(postProcSamples)}")
    # logger.debug(f"PostProcSamples type: {type(postProcSamples)}")
    # logger.debug(f"PostProcReferencesMetadata first 5:\n{postProcReferencesMetadata[:5]}")
    # logger.debug(f"PostProcReferencesMetadata last 5:\n{postProcReferencesMetadata[-5:]}")
    # logger.debug(f"PostProcReferencesMetadata length: {len(postProcReferencesMetadata)}")
    # logger.debug(f"PostProcReferencesMetadata type: {type(postProcReferencesMetadata)}")
    # logger.debug(f"PostProcReferences first 5:\n{postProcReferences[:5]}")
    # logger.debug(f"PostProcReferences last 5:\n{postProcReferences[-5:]}")
    # logger.debug(f"PostProcReferences length: {len(postProcReferences)}")
    # logger.debug(f"PostProcReferences type: {type(postProcReferences)}")

    # Create list of dataset metadata and samples
    datasetList = [samplesMetadata, allSamples]
    # datasetList = [postProcSamplesMetadata, postProcSamples]

    # Create list of reference metadata and data
    referenceList = [referencesMetadata, allReferences]
    # referenceList = [postProcReferencesMetadata, postProcReferences]

    # Write dataset
    with open(os.path.join(outputDirectory+'dataset-'+currentTime+'.json'), 'w') as dFileHandle:
        dFileHandle.write(json.dumps(datasetList))
    # Write reference
    with open(os.path.join(outputDirectory+'reference-'+currentTime+'.json'), 'w') as rFileHandle:
        rFileHandle.write(json.dumps(referenceList))


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
