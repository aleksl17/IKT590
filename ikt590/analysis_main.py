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
    inputDirectory='./signals/'
    timestampDataList = []
    valueDataList = []
    timeGapsList = []
    timeGapsMeanList = []
    timeGapsMedianList = []
    valueMeanList = []
    valueMedianList = []
    standardDeviationList = []
    varianceList = []
    totalRows = 0
    tmpX = 0
    tmpY = 0
    totalOutliers = 0
    
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
            logging.info(f"{fileName} - Highest Time Gap: {max(timeGaps)}")
            logging.info(f"{fileName} - Lowest Time Gap: {min(timeGaps)}")
            
            # Per file metadata
            timeGapsMean = numpy.mean(timeGaps)
            timeGapsMeanList.append(timeGapsMean)
            logging.info(f"{fileName} - Time Gap Mean: {timeGapsMean}")
            timeGapsMedian = numpy.median(timeGaps)
            timeGapsMedianList.append(timeGapsMedian)
            logging.info(f"{fileName} - Time Gap Median: {timeGapsMedian}")
            valueMax = max(valueData)
            logging.info(f"{fileName} - Highest Value: {valueMax}")
            valueMin = min(valueData)
            logging.info(f"{fileName} - Lowest Value: {valueMin}")
            valueMean = numpy.mean(valueData)
            valueMeanList.append(valueMean)
            logging.info(f"{fileName} - Value Mean: {valueMean}")
            valueMedian = numpy.median(valueData)
            valueMedianList.append(valueMedian)
            logging.info(f"{fileName} - Value Median: {valueMedian}")
            standardDeviation = numpy.std(valueData)
            standardDeviationList.append(standardDeviation)
            logging.info(f"{fileName} - Standard Deviation: {standardDeviation}")
            variance = numpy.var(valueData)
            varianceList.append(variance)
            logging.info(f"{fileName} - Variance: {variance}")
            outlierMultiplier = 3
            outliers = [value for value in valueData if value > outlierMultiplier*valueMedian]
            outliersNum = len(outliers)
            logging.info(f"{fileName} - Number of outliers with {outlierMultiplier}x multiplier: {outliersNum}")

            # Total metadata tmp
            tmpX += timeGapsMean * len(timeGaps)
            tmpY += len(timeGaps)
            totalRows += len(fileData.index)
            totalOutliers += outliersNum

    # Total metadata
    # totalTimeGapsMean = tmpX/tmpY
    totalTimeGapsMean = numpy.mean(timeGapsList)
    logging.info(f"Total Time Gap Mean: {totalTimeGapsMean}")
    totalTimeGapsMedian = numpy.median(timeGapsList)
    logging.info(f"Total Time Gap Median: {totalTimeGapsMedian}")
    totalValueMean = numpy.mean(valueDataList)
    logging.info(f"Total Value Mean: {totalValueMean}")
    totalValueMedian = numpy.median(valueDataList)
    logging.info(f"Total Value Median: {totalValueMedian}")
    logging.info(f"Total Rows: {totalRows}")
    logging.info(f"Total Outliers: {totalOutliers}")
    outlierPercentage = (totalOutliers/totalRows)*100
    logging.info(f"Outliers Percentage: {outlierPercentage}%")


if __name__ == "__main__":
    main()
