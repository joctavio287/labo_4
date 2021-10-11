import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd, os
# Es el path al directorio contenedor de ffts.py
path = "C:/repos/labo_4/YOUNG_DINAMICO/ffts.py"
os.chdir(path)
from ffts import *
os.chdir("C:/repos/labo_4/YOUNG_DINAMICO")
plt.rcParams['figure.figsize'] = [5, 5]

# Ploteo de las señales osci adquiridas por octi:
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (5, 5))
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
    tiempo = np.linspace(0, tiempo_total, len(df.datos))[comienzo:final]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    ax[i].plot(tiempo, datos)
fig.show()

# La que esta más buena (elegir el j):
j = 4
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/datita{}_osci.csv'.format(str(j)))
escala_temporal = 5e-1 # 500 ms cada cuadradito [s]
tiempo_total = escala_temporal*10

for k in range(len(df.datos)):
    if df.datos[k] == df.datos.max():
        comienzo = k

# El segundo parametro se ajusta a mano
final = comienzo + 2000
datos = df.datos[comienzo:final]# - np.mean(df.datos[comienzo:final])
tiempo = np.linspace(0, tiempo_total, len(df.datos))[comienzo:final]
tstep = (tiempo.max() - tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]
plt.figure('señal '+ str(j))
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
for i in range(12):
    picos[0]*i 

# De esta señal, me quedo con los picos positivos para calcularle el log y conseguir 
# el coef. de decaimiento. Para esto voy a usar la otra función que cree 'peaks':
picos_t, amplitud_t = peaks(tiempo = tiempo, señal = datos.tolist(), picos = True, labels = False)
plt.figure('picos_señal '+ str(j))
plt.scatter(picos_t, amplitud_t, s = 5)
# plt.yscale('log')
plt.show()

# Ahora voy a hacer un logplot de los datos y ajustar una lineal. La pendiente será 
# el coeficiente de decaimiento:

# Defino la función que calcula los parámetros (y sus varianzas):
def parametros(x, y, errores = False, errory = 0):
    N = len(x)
    DELTA = N*np.sum(x**2) - np.sum(x)**2
    a_1 = (np.sum(x**2)*np.sum(y) - np.sum(x)*np.sum(x*y))/DELTA
    a_2 = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/DELTA
    if errores == True:
        matriz_de_cov = np.array([[np.sum(x**2), -np.sum(x)],[-np.sum(x), N]])
        matriz_de_cov *= (errory**2/DELTA)
        return a_1, a_2, matriz_de_cov
    else:
        return a_1, a_2

# La uso y calculo dos tiras auxiliares para graficar el ajuste:
a_1, a_2, cov = parametros(np.array(picos_t), np.log(np.array(amplitud_t)), errores = True)
x_auxiliar = np.linspace(0, picos_t[-1] + 1, 1000) # tomo valores del 0 al 5 como pide el enunciado
y_auxiliar = a_1 + a_2*x_auxiliar

# Defino un texto en el cual voy a poner la matriz de covarianza:
 
v11, v12 = str(np.round(cov[0][0],2)), str(np.round(cov[0][1],2)) 
v21, v22 = str(np.round(cov[1][0],2)), str(np.round(cov[1][1],2))

# Grafico los resultados:
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1)#, figsize = (10, 10))
    # Los datos:
    ax.errorbar(picos_t, np.log(np.array(amplitud_t)), np.zeros(len(amplitud_t)),#np.full(len(y), .3),
                color = 'k',
                capsize = 4,
                fmt = '.', label = 'Datos')
    # El ajuste:
    ax.plot(x_auxiliar, y_auxiliar, color = 'red', label = 'Ajuste') 
    # ax.annotate(texto, (0.25, 0.25), size = 25)
    ax.legend(fontsize = 15)
    ax.set_xlim(0,picos_t[-1] + 1)
    # ax.set_ylim(0,5)
    fig.show()
    
df = pd.read_csv(r'C:\repos\labo_4\YOUNG_DINAMICO\Mediciones\medicio_osci4.csv')
aux = 100
tiempo, tension = df.tiempo[aux:], df.tension[aux:]

tstep = (tiempo.max() - tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]
plt.figure('señal '+ 'ruido')
plt.plot(tiempo, tension)
plt.show()

picos_ruido, altura_ruido = tirafft(tension, fsamp, log = True, picos = True,
threshold = None,
 prominence = (.9e-2, 50),
 height = None,
 distance = None,
 width = None,
 rel_height = None)

# Quedó buenisima, mirar las frecuencias en la transformada y los múltiplos acá
for i in range(12):
    picos[0]*i 
