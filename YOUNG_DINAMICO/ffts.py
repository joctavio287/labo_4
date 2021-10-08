import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.signal import find_peaks  
#===============================================================================
# 'tirafft' es una función para calcular la fft y graficarla, además encuentra los 
# picos de la señal (frecuencia y amplitud).
# Para usar utilizar el siguiente comando:
# import sys
# path = "C:/repos/labo_4/YOUNG_DINAMICO/fft_ejemplo.py"
# file = sys.path.append(path) 
# from ffts import *
#===============================================================================


def tirafft(señal,
 f_samp,
 formula:str = 'la señal',
 log = False,
 labels = True,
 picos = True,
 threshold = None,
 prominence = None,
 height = None,
 distance = None,
 width = None,
 rel_height = None
 ):
    '''
    INPUT: 
    señal: señal de entrada, preferentemente en formato np.array().

    f_samp: es la frecuencia de sampleo de la señal. i.e: 1/pasotemporal.

    formula: es una expresión en latex para printear en el gráfico.

    log: si la transformada la hace en escala logarítmica. Esto resulta conveniente
    para encontrar picos de frecuencias que tienen amplitud relativa muy baja.

    labels: si adhiere o no texto descriptivo en la imagen.

    picos: si buscar los picos de la FFT. Es útil, primero tomarla por False, para 
    poder verla en todo el rango y chequear, después, si al correrla en True no se 
    pierden picos.

    prominence: 'the minimum height necessary to descend to get from the summit to any
    higher terrain', si se pasan dos valores, el primer se interpreta como el mínimo valor;
    el segundo como el máximo.

    threshold: 'required vertical distance to its direct neighbouring samples. The first 
    element is always interpreted as the minimal and the second, if supplied, as the maximal
    required threshold.'

    height: 'required height of peaks. The first element is always interpreted as the minimal
    and the second, if supplied, as the maximal required height.'

    distance: 'required minimal horizontal distance (>= 1) in samples between neighbouring 
    peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining
    peaks.

    width: 'required width of peaks in samples. The first element is always interpreted as the
    minimal and the second, if supplied, as the maximal required width.

    rel_height: 'pass only if width is given. Chooses the relative height at which the peak 
    width is measured as a percentage of its prominence. 1.0 calculates the width of the peak
    at its lowest contour line while 0.5 evaluates at half the prominence height. Must be at 
    least 0.'

    OUTPUT:
    Si 'picos' =  True entonces devuelve una tupla compuesta de dos listas. La primera
    es la frecuencia donde estan los picos, la segunda son sus amplitudes.
    Por defecto, siempre se muestra la transformada
    '''
    señal_fft, N = np.fft.fft(señal), len(señal)
    xf = np.linspace(0, f_samp/2, int(N/2))
    yf = 2*np.abs(señal_fft[:N//2])/N

    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
        ax.plot(xf, yf, label = r'Transformada de Fourier de {}'.format(formula))
        if log == False:
            ax.set_ylabel('Amplitud espectral')
        else:
            ax.set_ylabel('Amplitud espectral (escala logaritmica)')
            ax.set_yscale('log')
        if picos == True:
            try:
                # picos_x, var_inservible = find_peaks(yf, prominence = prominence)
                picos_x, var_inservible = find_peaks(yf,
                threshold = threshold,
                prominence = prominence,
                height = height,
                distance = distance,
                width = width,
                rel_height = rel_height)
                ax.set_xlim(0, xf[picos_x[-1]] + 1)
                for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
                    ax.plot(x_p, y_p, marker = "o", markersize = 5,
                    label = 'Coordenadas del pico: ({}, {})'.format(np.round(x_p, 2), np.round(y_p, 6)))
            except:
                ax.set_xlim(0, f_samp/2)
        else: 
            ax.set_xlim(0, f_samp/2)
        ax.set_xlabel('Frecuencia [Hz]')
        if labels == True:
            ax.legend(fontsize = 12, loc = (.5, .7))
        fig.tight_layout()
        fig.show()
    if picos == True:
        return  [xf[x] for x in picos_x], [yf[x] for x in picos_x]