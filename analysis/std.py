import matplotlib.pyplot as plt
from math import floor, ceil
from turtle import circle
import numpy as np
import pandas
import os
# import ikt590.DataScripts.data_manipulation as data_manipulation

inputDirectory='./.localData/'

x = []

for file in os.listdir(inputDirectory):
        # Read files and interpolate
        csvData = pandas.read_csv(os.path.join(inputDirectory, file))
        x.extend(csvData['value'])

std = np.std(x)
print("Standard Deviation: ")
print(std)

mean = np.mean(x)
print("Mean:")
print(mean)

sorted = np.sort(x)
median = (sorted[floor(len(x)/2)] + sorted[ceil(len(x)/2)])/2
print("Median:")
print(median)

max = np.max(x)
print("Max: ")
print(max)

min = np.min(x)
print("Min: ")
print(min)

var = np.var(x)
print("Variance:")
print(var)

outliers = [xi for xi in x if xi > 3*mean]
print("Too high:")
print(len(outliers)/len(x))

#distribution
dist = np.zeros(50)
y = np.asarray(range(50))/2

for x1 in x:
    dist[round(x1*2)] = dist[round(x1*2)] + 1



# plt.plot(y, dist)
# plt.xticks()
# plt.show()