import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import logging
import math


def interpolation(data, sample_length = 15*60):
    """Interpolated data"""
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Find start time and end time of current data given as pandas DataFrame object
    start = datetime.strptime(data['timestamp'].iloc[0], '%Y-%m-%dT%H:%M:%S')
    end = datetime.strptime(data['timestamp'].iloc[-1], '%Y-%m-%dT%H:%M:%S')
    logger.debug(f"Start: {start}")
    logger.debug(f"End: {end}")
    interval = end - start
    logger.debug(f"Interval: {interval}")
    logger.debug(f"Interval total seconds: {interval.total_seconds()}")

    # Calculate samples based on total dataset length and desired sample length
    samples = math.floor(interval.total_seconds()/sample_length)
    logger.debug(f"Samples: {samples}")
    
    # Sets samples to be the same amount of original dataset
    # samples = len(data)
    # logger.debug(f"Samples: {samples}")

    # Convert timestamp to total seconds of x
    x = data['timestamp'].values.tolist()
    temp_x = []
    for t in x:
        interval = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - start
        temp_x.append(interval.total_seconds())
    x = temp_x
    logger.debug(f"Length of x: {len(x)}")
    y = data['value'].values.tolist()
    logger.debug(f"Length of y: {len(y)}")

    # Interpolate data
    f = interpolate.interp1d(x,y, fill_value='extrapolate')
    new_x = [(sample_length) * i for i in range(samples)]
    new_x = new_x[:len(x)]
    logger.debug(f"Length of new x: {len(new_x)}")
    new_y = f(new_x)
    new_y = new_y[:len(y)]
    logger.debug(f"Length of new y: {len(new_y)}")

    # return new_x, new_y
    return new_y
