from scipy import interpolate
from validating import validate_data
from get_time import get_time_intervals

def interpolation(x,y):
    f = interpolate.interp1d(x,y, fill_value='extrapolate')

    new_x = [(24*60*60/80)*i for i in range(80)]
    new_y = f(new_x)

    return new_y

def convert(r_json): #data is json
    size = len(r_json)

    x = []
    y = []

    for i in range(size):
        if r_json[i]['type'] == "VEvent" and not r_json[i]['sourceId'].split("/")[-1] == "hu":
            x.append(r_json[i]['timestamp'])
            y.append(float(r_json[i]['value']))


    x = get_time_intervals(x)

    new_y = interpolation(x,y)

    return new_y
    
    
