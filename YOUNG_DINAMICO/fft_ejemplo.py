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
# Importamos la función que creamos para condensar el trabajo:
#===============================================================================
import sys
path = "C:/repos/labo_4/YOUNG_DINAMICO/fft_ejemplo.py"
file = sys.path.append(path) 
from ffts import *
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