import logging
import pandas
import time
import os

import interpolate


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    
    for filename in os.listdir('./.localData'):
        csvData = pandas.read_csv(f"./.localData/{filename}")
        # print(csvData)
        data = interpolate.interpolation(csvData)

    print('Hello, World!')

if __name__ == "__main__":
    main()