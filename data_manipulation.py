import pandas

def manipulate_data(file, interval=1440):
    data = pandas.read_csv(file)
    