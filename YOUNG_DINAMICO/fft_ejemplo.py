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

V = 2*np.sin(2*np.pi*f0*t) + 1*np.sin(2*np.pi*f1*t) 

# Graficamos la función de prueba
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    ax.plot(t, V)
    ax.set_ylabel('Amplitud')
    ax.set_xlabel('Tiempo [s]')
    plt.tight_layout()
    plt.show()

# Hacemos la trasnformada de fourier (FFT) y graficamos.
fft = np.fft.fft(V)

# Calular el vector de frecuencias. Se calcula hasta la mitad porque
# el algoritmo espeja los valores (para entender descomentar frecs2
# y las lineas dentro del plot).

frecs = np.linspace(0, fsamp/2, int(N/2)) 
# frecs2 = np.linspace(0, fsamp, N)

# Graficamos la tasnformada de fourier
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    # PREGUNTAR POR QUE HAY QUE NORMALIZAR CON UN MEDIO. Es algo del teo del muestreo,
    # como que para definir una seña periódica necesitas como minimos dos puntos
    ax.plot(frecs, 2*np.abs(fft[:N//2])/N) 
    # ax.plot(frecs2, 2*np.abs(fft)/N) 
    # ax.set_xlim([0,1000])
    ax.set_xlim([0, 25])
    ax.set_ylabel('Amplitud espectral')
    ax.set_xlabel('Frecuencia [Hz]')
    plt.tight_layout()
    plt.show()

 
# Función para graficar transformada
def tirafft(señal, f_samp, formula:str = 'la señal', log = False, picos = True, prominence = 1):
    '''
    INPUT: 
    señal: señal de entrada, preferentemente en formato np.array().
    f_samp: es la frecuencia de sampleo de la señal. i.e: 1/pasotemporal.
    formula: es una expresión en latex para printear en el gráfico.
    log: si la transformada la hace en escala logarítmica. Esto resulta conveniente
    para encontrar picos de frecuencias que tienen amplitud relativa muy baja.
    picos: si buscar los picos de la FFT. Es útil, primero tomarla por False, para poder
    verla en todo el rango y chequear, después, si al correrla en True no se pierden picos.
    prominence: es la altura del menor pico, si se pasan dos valores, el segundo se 
    interpreta como el maximo valor.
    '''
    señal_fft, N = np.fft.fft(señal), len(señal)
    xf = np.linspace(0, f_samp/2, int(N/2))
    yf = 2*np.abs(señal_fft[:N//2])/N
    if picos == True:
        picos_x, intensidad_picos = find_peaks(yf, prominence = prominence)
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
        ax.plot(xf, yf, label = r'Transformada de Fourier de {}'.format(formula))
        if log == False:
            ax.set_ylabel('Amplitud espectral')
        else:
            ax.set_ylabel('Amplitud espectral (escala logaritmica)')
            ax.set_yscale('log')
        if picos == True:
            ax.set_xlim(0, xf[picos_x[-1]] + 1)
            for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
                 ax.plot(x_p, y_p, marker = "o", markersize = 5,
                 label = 'Coordenadas del pico: ({}, {})'.format(np.round(x_p, 2), np.round(y_p, 4)))
        else: 
            ax.set_xlim(0, f_samp/2)
        ax.set_xlabel('Frecuencia [Hz]')
        ax.legend(fontsize = 15)
        plt.tight_layout()
        plt.show()
    if picos == True:
        return  [xf[x] for x in picos_x], [yf[x] for x in picos_x]
        
        
picos, alturas = tirafft(V, fsamp, str(r'$sin(2\, \pi\, {} \, t) +sin(2\, \pi\, {} \, t)$'.format(f0,f1)))
V2 = 2*np.sin(2*np.pi*f0*t) + .001*np.sin(2*np.pi*f1*t) 
picos, altura = tirafft(V2, fsamp, log = True, prominence =.0001)
# len(str(prominence).split('.')[1]))))