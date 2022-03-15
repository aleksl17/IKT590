from datetime import datetime, timedelta
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
    # purgeIndexList = []
    perFileSampleLengthList = []
    referencesMaxList = []
    postProcSamplesMetadata = []
    postProcSamples = []
    postProcReferencesMetadata = []
    postProcReferences = []
    invalidSamplecounter = 0

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
            tmpReferenceList = []
            tmpReferenceMetadataList = []
            for i in range(len(valueData)-sample_size):
                referenceList.append(valueData[i:i+sample_size])
                referencesMetadata.append([file.split('.csv')[0], str(refTime), valueData[i]])
                refTime = refTime + timedelta(minutes=15)
            refMax = numpy.mean(referenceList)*outlierMulti
            for i, reference in enumerate(referenceList):
                refMean = numpy.mean(reference)
                if refMean < refMax:
                    tmpReferenceList.append(reference)
                    tmpReferenceMetadataList.append(referencesMetadata[i])
            postProcReferencesMetadata.extend(tmpReferenceMetadataList)
            postProcReferences.extend(tmpReferenceList)
            # references.extend(referenceList)
            # referencesMaxList.append(outlierMulti*numpy.median(referenceList))
            # print(referencesMaxList)
            
            # Interpolate data and convert to python list
            intData = data_interpolate.interpolation(csvData)
            intData = intData.tolist()
            
            # Normalize data
            normData = data_normalize.normalize(intData)

            # Split data into samples with overlap and create respective metadata tabel
            sampleList = []
            tmpSampleList = []
            tmpSampleMetadataList = []
            for set in range(len(normData)-sample_size):
                sampleList.append(normData[set:set+sample_size])
                samplesMetadata.append([file.split('.csv')[0], csvData['timestamp'][set], normData[set]])
            samMax = numpy.mean(sampleList)*outlierMulti
            for i, sample in enumerate(sampleList):
                samMean = numpy.mean(sample)
                if samMean < samMax:
                    tmpSampleList.append(sample)
                    tmpSampleMetadataList.append(samplesMetadata[i])
                else:
                    invalidSamplecounter += 1
            postProcSamplesMetadata.extend(tmpSampleMetadataList)
            postProcSamples.extend(tmpSampleList)

                    
            # perFileSampleLengthList.append(len(sampleList))
            # print(perFileSampleLengthList)
    
    # logger.debug(f"references: {references[0]}")
    # logger.debug(f"samples: {samples[0]}")

    logger.info(f"Invalid Samples removed: {invalidSamplecounter}")

    logger.debug(f"referencesMaxList: {referencesMaxList}")
    logger.debug(f"perFileSampleLengthList: {perFileSampleLengthList}")
    logger.debug(f"referencesMetadata length: {len(referencesMetadata)}")
    logger.debug(f"references length: {len(references)}")
    # logger.debug(f"references range length: {range(len(references))}")
    # logger.debug(f"references shape: {numpy.array(references, dtype=object).shape}")
    # logger.debug(f"references shape: {len(references[0])}")
    # logger.debug(f"references shape: {len(references[0][0])}")
    # logger.debug(f"references shape: {len(references[0][0][0])}")
    # logger.debug(f"references shape: {len(references[0][0][0][0])}")
    # logger.debug(f"references shape: {len(references[0][0][0][0][0])}")
    logger.debug(f"samplesMetadata length: {len(samplesMetadata)}")
    logger.debug(f"samples length: {len(samples)}")
    
    # Only appends wanted samples to dataset
    # for rdx in range(len(referencesMetadata)):
    #     if rdx % 1000 == 0:
    #         print(rdx)
    #     refMean = numpy.mean(references[rdx])
    #     if rdx < perFileSampleLengthList[0]:
    #         refMax = referencesMaxList[0]
    #     elif rdx < perFileSampleLengthList[1]:
    #         refMax = referencesMaxList[1]
    #     elif rdx < perFileSampleLengthList[2]:
    #         refMax = referencesMaxList[2]
    #     elif rdx < perFileSampleLengthList[3]:
    #         refMax = referencesMaxList[3]
    #     # print(refMean)
    #     # print(refMax)
    #     if refMean < refMax:
    #         PostProcSamplesMetadata.append(samplesMetadata[rdx])
    #         PostProcSamples.append(samples[rdx])
    #         PostProcReferencesMetadata.append(referencesMetadata[rdx])
    #         PostProcReferences.append(references[rdx])

    # for idx, rf in enumerate(referencesMaxList):
    #     print(f"idx: {idx}")
    #     refMax = referencesMaxList[idx]
    #     prevLen = 0
    #     for smpl in range(perFileSampleLengthList[idx]+prevLen):
    #         refMean = numpy.mean(samples[crrntSmpl])
    #         if refMean < refMax:
    #             PostProcSamplesMetadata.append(samplesMetadata[crrntSmpl])
    #             PostProcSamples.append(samples[crrntSmpl])
    #             PostProcReferencesMetadata.append(referencesMetadata[crrntSmpl])
    #             PostProcReferences.append(references[crrntSmpl])
    #         prevLen = perFileSampleLengthList[idx]
    #     crrntSmpl = smpl+prevLen
    #     print(f"smpl: {crrntSmpl}")


    logger.debug(f"PostProcSamplesMetadata first 5:\n{postProcSamplesMetadata[:5]}")
    logger.debug(f"PostProcSamplesMetadata last 5:\n{postProcSamplesMetadata[-5:]}")
    logger.debug(f"PostProcSamplesMetadata length: {len(postProcSamplesMetadata)}")
    logger.debug(f"PostProcSamplesMetadata type: {type(postProcSamplesMetadata)}")
    logger.debug(f"PostProcSamples first 5:\n{postProcSamples[:5]}")
    logger.debug(f"PostProcSamples last 5:\n{postProcSamples[-5:]}")
    logger.debug(f"PostProcSamples length: {len(postProcSamples)}")
    logger.debug(f"PostProcSamples type: {type(postProcSamples)}")
    logger.debug(f"PostProcReferencesMetadata first 5:\n{postProcReferencesMetadata[:5]}")
    logger.debug(f"PostProcReferencesMetadata last 5:\n{postProcReferencesMetadata[-5:]}")
    logger.debug(f"PostProcReferencesMetadata length: {len(postProcReferencesMetadata)}")
    logger.debug(f"PostProcReferencesMetadata type: {type(postProcReferencesMetadata)}")
    logger.debug(f"PostProcReferences first 5:\n{postProcReferences[:5]}")
    logger.debug(f"PostProcReferences last 5:\n{postProcReferences[-5:]}")
    logger.debug(f"PostProcReferences length: {len(postProcReferences)}")
    logger.debug(f"PostProcReferences type: {type(postProcReferences)}")












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
    datasetList = [postProcSamplesMetadata, postProcSamples]
    # logger.debug(f"datasetList length: {len(datasetList)}")
    # logger.debug(f"datasetList type: {type(datasetList)}")
    # logger.debug(f"datasetList shape: {len(datasetList[0])}")
    # logger.debug(f"datasetList shape: {len(datasetList[1])}")
    # logger.debug(f"datasetList shape: {len(datasetList[1][0])}")
    # logger.debug(f"datasetList shape: {len(datasetList[0][1])}")
    # logger.debug(f"datasetList shape: {len(datasetList[0][0][0])}")
    # logger.debug(f"datasetList shape: {len(datasetList[0][0][0])}")
    # logger.debug(datasetList)
    # Create list of reference metadata and data
    referenceList = [postProcReferencesMetadata, postProcReferences]

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
