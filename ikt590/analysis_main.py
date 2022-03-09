from datetime import datetime
import logging
import pandas
import numpy
import time
import os


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    
    # Variables
    inputDirectory='./.tmpData/'
    timestampDataList = []
    valueDataList = []
    timeGapsList = []
    meanList = []
    medianList = []
    standardDeviationList = []
    varianceList = []
    totalRows = []
    tmpX = 0
    tmpY = 0
    
    for file in os.listdir(inputDirectory):
        if file.endswith('.csv'):
            # Read files and declare values
            fileName = file.split('.csv')[0]
            fileData = pandas.read_csv(os.path.join(inputDirectory, file))
            timestampData = fileData['timestamp']
            timestampDataList.extend(timestampData)
            valueData = fileData['value']
            valueDataList.extend(valueData)
            
            # Find datetime gaps
            timeGaps = []
            for i in range(len(timestampData)-1):
                start = time.mktime(datetime.strptime(timestampData[i],'%Y-%m-%dT%H:%M:%S').timetuple())
                end = time.mktime(datetime.strptime(timestampData[i+1],'%Y-%m-%dT%H:%M:%S').timetuple())
                timeGaps.append(end - start)
            timeGapsList.extend(timeGaps)
            logging.info(f"{fileName} - Highest time gap: {max(timeGaps)}")
            logging.info(f"{fileName} - Lowest time gap: {min(timeGaps)}")
            
            # Per file metadata
            mean = numpy.mean(timeGaps)
            meanList.append(mean)
            logging.info(f"{fileName} - Mean: {mean}")
            median = numpy.median(timeGaps)
            medianList.append(median)
            logging.info(f"{fileName} - Median: {mean}")
            standardDeviation = numpy.std(valueData)
            standardDeviationList.append(standardDeviation)
            logging.info(f"{fileName} - Standard Deviation: {standardDeviation}")
            variance = numpy.var(valueData)
            varianceList.append(variance)
            logging.info(f"{fileName} - Variance: {variance}")
            outlierMultiplier = 3
            outliers = [row for row in valueData if valueData > outlierMultiplier*median]
            outliersNum = len(outliers)/len(valueData)
            logging.info(f"{fileName} - Number of outliers with {outlierMultiplier}x multiplier: {outliersNum}")

            # Total metadata tmp
            tmpX += mean * len(timeGaps)
            tmpY += len(timeGaps)
            totalRows.append(len(fileData.index))

    # Total metadata
    totalMean = tmpX/tmpY
    logging.info(f"Total mean: {totalMean}")
    totalRowsSum = numpy.sum(totalRows)
    logging.info(f"Total rows: {totalRowsSum}")


if __name__ == "__main__":
    main()
