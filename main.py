import logging
import time
import os

import import_data


def main():
    # Initialize logging
    if not os.path.exists('./.logs'):
        os.makedirs('./.logs')
    getTime = str(int(time.time()))
    logFile = os.path.join('./.logs/' + getTime + '.log')
    logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    
    logging.info('Started')
    ################ ======== Functions/code goes below here ======== ################

    # import_data.import_data(dataDirectory='.tdata')
    import_data.import_data()

    ################ ======== Functions/code goes above here ======== ################
    logging.info('Finished')


if __name__ == "__main__":
    main()