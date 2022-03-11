import logging
import pandas
import numpy
import time
import json
import os

import helpers.data_interpolate as data_interpolate
import helpers.data_normalize as data_normalize

# TODO:
# Optimization: Use Numpy arrays instead of Python lists
# NOTE
# Reference and dataset might not align due to time-shift from interpolation.
# Noen som bør sjekkes ut:
# Hvis man .poper en liste i en for loop, vil da indexes offsetes med -1?


def create_dataset(inputDirectory='./.tmpData/', outputDirectory='./.tmpData/', sample_size=40, outlierMulti=3):
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
    samples = []
    references = []
    referencesMetadata = []
    purgeIndexList = []
    perFileSampleLengthList = []
    referencesMaxList = []

    # Creates list of DataFrames from data from CSV files input directory
    for file in os.listdir(inputDirectory):
        if file.endswith('.csv'):
            # Read files
            csvData = pandas.read_csv(os.path.join(inputDirectory, file))
            valueData = csvData['value'].tolist()
            logger.debug(f"Read file: {file}")

            # Create reference "samples"
            referenceList = []
            for i in range(len(valueData)-sample_size):
                referenceList.append(valueData[i:i+sample_size])
                references.extend(referenceList)
                referencesMetadata.append([file.split('.csv')[0], csvData['timestamp'][i], valueData[i]])
            referencesMaxList.append(outlierMulti*numpy.median(referenceList))
            print(referencesMaxList)
            
            # Interpolate data and convert to python list
            intData = data_interpolate.interpolation(csvData)
            intData = intData.tolist()
            
            # Normalize data
            normData = data_normalize.normalize(intData)

            # Split data into samples with overlap and create respective metadata tabel
            sampleList = []
            for set in range(len(normData)-sample_size):
                sampleList.append(normData[set:set+sample_size])
                samples.extend(sampleList)
                samplesMetadata.append([file.split('.csv')[0], csvData['timestamp'][set], normData[set]])
            perFileSampleLengthList.append(len(sampleList))
            print(perFileSampleLengthList)

            # Create list of datasets
            # datasetList.append([metadata, samples])





    # # Find indexes of "samples" to be removed from reference and dataset
    # referenceMax = outlierMulti*numpy.median(reference)
    # logger.info(f"Outliers max value: {referenceMax}")
    # for idx, ref in enumerate(reference):
    #     refMean = numpy.mean(ref)
    #     # print(f"ref: {ref}")
    #     # print(f"refMean: {refMean}")
    #     if refMean > referenceMax:
    #         print(f"idx: {idx}")
    #         print(f"refMean: {refMean}")
    #         print(f"Substring to be purged: {reference[idx]}")
    #         purgeIndexList.append(idx)

    # # Purge unwanted outliers
    # logger.debug(f"Purge Indexes: {purgeIndexList}")
    # logger.debug(f"Indexes to be purged: {len(purgeIndexList)}")
    # logger.debug(f"Lengths before purge: {len(reference), len(referenceMetadata), len(samples), len(datasetMetadata)} - All lengths should be the same")
    # for iterIndex in range(len(reference)):
    #     if iterIndex in purgeIndexList:
    #         # print(f"Purging: {reference[iterIndex]}")
    #         # print(f"Where mean SHOULD be: {numpy.mean(reference[iterIndex])}")
    #         reference.pop(iterIndex)
    #         referenceMetadata.pop(iterIndex)
    #         samples.pop(iterIndex)
    #         datasetMetadata.pop(iterIndex)
    # logger.debug(f"Lengths after purge: {len(reference), len(referenceMetadata), len(samples), len(datasetMetadata)} - All lengths should be the same")
    # logger.info(f"Samples purged: {len(purgeIndexList)}")




    # Create list of dataset metadata and samples
    datasetList = [samplesMetadata, samples]
    # Create list of reference metadata and data
    referenceList = [referencesMetadata, references]

    # logger.debug(f"datasetList first:\n{datasetList[:1]}")
    # logger.debug(f"referenceList first:\n{referenceList[:1]}")

    # Create ouput directory
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    # Write dataset
    with open(os.path.join(outputDirectory+'dataset-'+currentTime+'.json'), 'w') as filehandle:
        filehandle.write(json.dumps(datasetList))
    # Write reference
    with open(os.path.join(outputDirectory+'reference-'+currentTime+'.json'), 'w') as filehandle:
        filehandle.write(json.dumps(referenceList))


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
