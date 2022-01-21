from scipy import interpolate
import logging
import math
from datetime import datetime
import matplotlib.pyplot as plt


def interpolation(data, sample_length = 15*60):
    """Interpolated data"""
    # Initialize logger
    logger = logging.getLogger(__name__)
    print('Now in interpolate')

    # data = data[:50]
    
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

    # Expected x
    # x = [0, 313, 921, 921, ...]

    x = []
    y = []
    x = data['timestamp'].values.tolist()

    temp_x = []

    for t in x:
        interval = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - start
        temp_x.append(interval.total_seconds())

    x = temp_x

    logger.debug(f"Length of x: {len(x)}")
    logger.debug(f"First index of x: {x[0]}")
    y = data['value'].values.tolist()
    logger.debug(f"Length of y: {len(y)}")
    # logger.debug(type(y[0]))
    # logger.debug(y)

    f = interpolate.interp1d(x,y, fill_value='extrapolate')
    logger.debug(f"f: {f}")
    new_x = [(sample_length) * i for i in range(samples)]
    logger.debug(f"new x: {new_x}")
    logger.debug(f"Length of new x: {len(new_x)}")

    # for row in x:
    #     if x[:-1] >= new_x[:-1]:
    #         x.pop()
    # logger.debug(f"x after pop: {x}")
    # logger.debug(f"Length of x after pop: {len(x)}")

    new_y = f(new_x)
    logger.debug(f"new y: {new_y}")

    # plt.plot(new_y)
    # plt.show()
    return new_y
