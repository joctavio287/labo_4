import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.signal import find_peaks     
#===============================================================================
# Entendiendo la fft. Generamos una función de prueba. 
# El tiempo está en segundos y la frecuencia en Hertz.
#===============================================================================

N = 10000 # cantidad de pasos
tmax = 10 # tiempo máximo
tstep = tmax/N # paso temporal
fsamp = 1/tstep # frecuencia de sampleo
t = np.arange(0, tmax, tstep)

# Una función de prueba.
f0, f1 = 7, 20
V = 2*np.sin(2*np.pi*f0*t) + 1*np.sin(2*np.pi*f1*t) 

# Graficamos la función de prueba
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5), num = 0)
    ax.plot(t, V)
    ax.set_ylabel('Amplitud')
    ax.set_xlabel('Tiempo [s]')
    fig.tight_layout()
    fig.show()

# Hacemos la trasnformada de fourier (FFT) y graficamos.
fft = np.fft.fft(V)

#===============================================================================
# Calular el vector de frecuencias. Se calcula hasta la mitad porque 
# el algoritmo espeja los valores (para entender descomentar frecs2 
# y las lineas dentro del plot).
#===============================================================================
frecs = np.linspace(0, fsamp/2, int(N/2)) 

# frecs2 = np.linspace(0, fsamp, N)
plt.figure
# Graficamos la tansformada de fourier
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1,
     figsize = (10, 5),
     num = 1)
    # PREGUNTAR POR QUE HAY QUE NORMALIZAR CON UN MEDIO. Es algo del teo del muestreo,
    # como que para definir una seña periódica necesitas como minimos dos puntos
    ax.plot(frecs, 2*np.abs(fft[:N//2])/N) 
    # ax.plot(frecs2, 2*np.abs(fft)/N) 
    # ax.set_xlim([0,1000])
    ax.set_xlim([0, 25])
    ax.set_ylabel('Amplitud espectral')
    ax.set_xlabel('Frecuencia [Hz]')
    fig.tight_layout()
    fig.show()