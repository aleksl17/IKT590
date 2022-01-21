from scipy import interpolate
from datetime import datetime
import logging

# interval = 24 #interval in hours
samples = 80 #amount of samples per timestamp

def interpolate(data):
    """Interpolated data"""

    start = data['timestamp'].iloc[0] # First column of first row in pandas DataFrame
    start = datetime.strptime(start, )
    end = data['timestamp'].iloc[-1] # First column for last row in pandas DataFrame

    # interval = data.iloc[0][0] - data[-1][0]
    interval = start - end
    # Initalize logger
    logger = logging.getLogger(__name__)
    
    # x = timestamps
    x = []
    # y = measured values
    y = []
    for d in data:
        x.append(d[0])
        y.append(d[1])


        
    
    f = interpolate(x,y, fill_value='extrapolate')
    new_x = [(interval * 60 * 60 / samples) * i for i in range(samples)]
    new_y = f(new_x)

    return new_y

