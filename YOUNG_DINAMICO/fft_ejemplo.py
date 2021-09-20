# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks     

# Generamos una función de prueba. El tiempo está en segundos y la frecuencia en Hertz.

N = 10000 # cantidad de pasos
tmax = 10 # tiempo máximo
tstep = tmax/N # paso temporal
fsamp = 1/tstep # frecuencia de sampleo
t = np.arange(0, tmax, tstep)

# Una función de prueba.
f0, f1 = 7, 20

V = np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f1*t) 

# Graficamos la función de prueba
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    ax.plot(t, V)
    ax.set_ylabel('Potencia')
    ax.set_xlabel('Tiempo [s]')
    plt.tight_layout()
    plt.show()

# Hacemos la trasnformada de fourier (FFT) y graficamos.
fft = np.fft.fft(V)

# Calular el vector de frecuencias.
frecs = np.linspace(0, fsamp/2, int(N/2))

# Graficamos la tasnformada de fourier
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    ax.plot(frecs, 2*np.abs(fft[:N//2])/N) # PREGUNTAR POR QUE HAY QUE NORMALIZAR CON UN MEDIO
    ax.set_xlim([0, 25])
    ax.set_ylabel('Potencia')
    ax.set_xlabel('Frecuencia [Hz]')
    plt.tight_layout()
    plt.show()

 
# Función para graficar transformada
def tirafft(señal, f_samp, formula):
    señal_fft, N = np.fft.fft(señal), len(señal)
    xf = np.linspace(0, f_samp/2, int(N/2))
    yf = 2*np.abs(señal_fft[:N//2])/N
    picos_x, intensidad_picos = find_peaks(yf, height = 1)
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
        ax.plot(xf, yf)
        ax.set_xlim(0, xf[picos_x[-1]] + 1)
        ax.set_ylabel('Potencia')
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_title(r'FFT de {}'.format(formula))
        plt.tight_layout()
        plt.show()
        
    return xf[picos_x], intensidad_picos['peak_heights']

tirafft(V, fsamp, str(r'$sin(2\, \pi\, {} \, t) +sin(2\, \pi\, {} \, t)$'.format(f0,f1)))

























