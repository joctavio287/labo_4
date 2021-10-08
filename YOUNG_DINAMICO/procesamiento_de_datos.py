import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd, sys
# Es el path al directorio contenedor de fft_ejemplo.py
file = sys.path.append("C:/repos/labo_4/YOUNG_DINAMICO/ffts.py") 
from ffts import *

# Ploteo de las señales osci adquiridas por octi:
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (10,10))
ax = ax.flatten()
for i in range(6):
    df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/datita{}_osci.csv'.format(str(i)))
    escala_temporal = 5e-1 # 500 ms cada cuadradito [s]
    tiempo_total = escala_temporal*10
    for k in range(len(df.datos)):
        if df.datos[k] == df.datos.max():
            comienzo = k
    final = comienzo + 1800
    datos = df.datos[comienzo:final] #- np.mean(df.datos[comienzo:final])
    tiempo = np.linspace(0, tiempo_total, len(datos))
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    ax[i].plot(tiempo, datos)
fig.show()

# La que esta más buena (elegir el i):
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/datita{}_osci.csv'.format(str(5)))
escala_temporal = 5e-1 # 500 ms cada cuadradito [s]
tiempo_total = escala_temporal*10

for k in range(len(df.datos)):
    if df.datos[k] == df.datos.max():
        comienzo = k
final = comienzo + 1800
datos = df.datos[comienzo:final]# - np.mean(df.datos[comienzo:final])
tiempo = np.linspace(0, tiempo_total, len(datos))
tstep = (tiempo.max() - tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]
plt.figure(0)
plt.plot(tiempo, datos)
plt.show()

picos, altura = tirafft(datos, fsamp, log = True, picos = True,
threshold = None,
 prominence = (.9e-2, 50),
 height = None,
 distance = None,
 width = None,
 rel_height = None)

# Quedó buenisima, mirar las frecuencias en la transformada y los múltiplos acá
for i in range(10):
    picos[0]*i 

# De esta señal, me quedo con los picos positivos para calcularle el log y conseguir 
# el coef. de decaimiento. Para esto voy a usar la otra función que cree 'peaks':
picos_t, amplitud_t = peaks(tiempo = tiempo, señal = datos.tolist(), picos = True, labels = False)

# Ahora voy a hacer un logplot de los datos y ajustar una lineal. La pendiente será 
# el coeficiente de decaimiento:


