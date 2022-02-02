import logging
import time
import os

import data_import
import data_manipulation


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    currentTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + currentTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s %(name)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    
    logging.info('Started')
    ################ ======== Functions/code goes below here ======== ################
    
    # Download data via API if .localData folder is empty
    #import_data.import_data(signalFrom="2021-01-01T01:00:00.000Z", signalTo="2022-01-01T01:00:00.000Z")
    idd = data_import.import_data()
    print(idd)

    # Create dataset from local data files
    #for file in os.listdir('./.localData'):
    #    data_manipulation.manipulate_data(f"./.localData/{file}")
    
    ################ ======== Functions/code goes above here ======== ################
    logging.info('Finished')


if __name__ == "__main__":
    main()
