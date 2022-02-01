import csv
import numpy as np

def get_data():
    alldata= []

    with open('New_York_Hourly.csv', newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        for line in file:
            alldata.append(float(line[2]))


    #sample
    dataset = []
    sample_size = 24
    data_size = len(alldata)

    for i in range(round(data_size/sample_size)):
        dataset.append(alldata[i*sample_size:(i+1)*sample_size])
    
    return dataset

