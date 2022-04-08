import logging
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import time
import pandas

def fourier_transform(input_data):
    """Perform fourier transform on data"""

    # Initalize logger
    logger = logging.getLogger(__name__)

    x = list(range(len(input_data.index[0:100])))
    y = input_data.value[0:100]

    # Plot input_data
    fig = plt.figure()
    plt.plot(x, y)
    plt.ylabel('Value')
    plt.xlabel('Time')
    # plt.show()

    # Perform fast fourier transform
    f = abs(numpy.fft.fft(y))

    # No idea what we do here
    num = numpy.size(x)
    freq = [i / num for i in list(range(num))]
    spectrum = f.real*f.real+f.imag*f.imag
    nspectrum = spectrum/spectrum[0]

    # Plot FFT graph
    plt.semilogy(freq, nspectrum)
    # plt.show()

    # Decompose and plot data
    res = sm.tsa.seasonal_decompose(list(input_data['value'][0:100]), period=1)
    res.plot()
    plt.show()