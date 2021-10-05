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

 
#===============================================================================
# En base a lo de arriba creo una función para calcular la fft y graficarla,
# encontrando los picos (frecuencia y amplitud).
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

#===============================================================================
# Algunos ejemplos de uso de la función.
#===============================================================================

# Prueba        
picos, altura = tirafft(V, fsamp, str(r'$sin(2\, \pi\, {} \, t) + sin(2\, \pi\, {} \, t)$'.format(f0, f1)), prominence = 1)

# Señal que potencialmente esconde picos
V2_ = 2*np.sin(2*np.pi*f1*t)*np.exp(-t*1/20) + .01*np.sin(2*np.pi*f0*t)
V2 = sp.norm.rvs(V2_, scale = .01, random_state = 15) 

# Si corremos así nomas, nos comemos un pico. Cuidado que si no se tira ningun parametro va 
# a encontrar TODOS los picos:
tirafft(V2, fsamp, picos = True, labels = False)

# Con escala logarítmica, para encontrar picos muy disminuidos:
tirafft(V2, fsamp, picos = True, log = True, prominence = 1)

# Ahora que se ven los picos se puede intuir que el más chico tiene una
# amplitud aproximada de 10^-3:

picos2, altura2 = tirafft(V2, fsamp, log = True, picos = True, labels = True,
 threshold = None,
 prominence = (1e-3, 2),
 height = None,
 distance = None,
 width = None,
 rel_height = None)
 
# 'datita4.csv' es el bueno
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/datita{}.csv'.format(str(4)))
aux = 550
tiempo = np.linspace(0, .5, len(df.datos))[aux:2000]
tstep = (tiempo.max()-tiempo.min())/len(tiempo)
fsamp = 1/tstep
plt.figure(0)
plt.plot(np.linspace(0,.5,len(df.datos))[aux:2000], df.datos[aux:2000])
plt.show()

picos, altura = tirafft(df.datos[aux:2000].tolist(), fsamp, log = True, picos = True,
threshold = None,
 prominence = (.9e-2, 50),
 height = None,
 distance = None,
 width = None,
 rel_height = None)

for i in range(10):
    picos[0]*i