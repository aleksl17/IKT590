import os
import pandas
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

inputDirectory='./.localData/'

x = []
sX = []
tempX = 0
tempY = 0
total_time_gaps = []

for file in os.listdir(inputDirectory):
        # Read files and interpolate
        csvData = pandas.read_csv(os.path.join(inputDirectory, file))
        sX = csvData['timestamp']
        x.extend(sX)
        
        time_gaps = []
        
        for i in range(len(sX)-1):
            start = time.mktime(datetime.strptime(sX[i],'%Y-%m-%dT%H:%M:%S').timetuple())
            end = time.mktime(datetime.strptime(sX[i+1],'%Y-%m-%dT%H:%M:%S').timetuple())
            time_gaps.append(end - start)

        mean = np.mean(time_gaps)
        median = np.median(time_gaps)
        tempX += mean * len(time_gaps)
        tempY += len(time_gaps)
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(max(time_gaps))

        total_time_gaps.extend(time_gaps)

# onePercentile = np.percentile(total_time_gaps, 1)
# tenthPercentile = np.percentile(total_time_gaps, 10)
# nintiethPercentile = np.percentile(total_time_gaps, 90)
# nintieninethPercentile = np.percentile(total_time_gaps, 99)
# print(f"Top 99%: {onePercentile}")
# print(f"Top 90%: {tenthPercentile}")
# print(f"Top 10%: {nintiethPercentile}")
# print(f"Top 1%: {nintieninethPercentile}")

print(max(total_time_gaps))
print(f"Total Mean: {tempX / tempY}")


boundry = 110
distData = np.zeros(boundry)
distY = np.asarray(range(boundry))*10

for item in total_time_gaps:
    tmpX = round(item/10)
    # print(tmpX)
    if tmpX < boundry:
        distData[tmpX] = distData[tmpX] + 1

plt.plot(distY[80:], distData[80:])
plt.show()
