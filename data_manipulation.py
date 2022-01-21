import logging
import pandas
import numpy

import interpolate


def manipulate_data(file, interval=1440):
    """Manipulates data"""
    
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    # Variables
    x = numpy.array([])
    y = numpy.array([])

    data = pandas.read_csv(file)
    print(data)
    print(type(data))

    interpolate.interpolate(data)

    print(data)
    