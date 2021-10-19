import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd, os
# Es el path al directorio contenedor de ffts.py
path = "C:/repos/labo_4/YOUNG_DINAMICO/"
os.chdir(path)
from ffts import *
os.chdir("C:/repos/labo_4/YOUNG_DINAMICO")

# =============================================================================
# Armo una clase para hacer regresiones lineales unidimensionales. Si después
# hacemos regresiones más chetas podemos retocarla:
# =============================================================================
class regresion_lineal:

    def __init__(self) -> None:
        self.parametros = None
        self.cov_parametros = None
        self.r = None
        self.x = None
        self.y = None
        pass

    def fit(self, x, y, cov_y = None, ordenada = False):
        '''
        INPUT: (x, y) son los datos para ajustar; 'cov_y' es la matriz de covarianza de los datos, de no haber errores, por defecto es la identidad;
        'ordenada' es por si se quiere o no tener como output la ordenada.
        OUTPUT: actualiza los coeficientes del ajuste y su matriz de covarianza.
        '''
        self.x, self.y = x, y
        if cov_y is None:
            cov_y = np.diag(np.ones(y.shape))
        #Esta hecha con matrices por si después queremos ampliarla
        A = [x]
        if ordenada == True:
            A = [np.ones(x.shape)] + A
        A = np.matrix(A).T # Matriz de vandermonte
        inversa_cov = np.linalg.inv(cov_y)
        parametros = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.T,inversa_cov),A)), A.T), inversa_cov),y)
        cov_parametros = np.linalg.inv(np.dot(A.T, np.dot(inversa_cov,A)))
        self.parametros, self.cov_parametros = np.array(parametros)[0], np.array(cov_parametros)
    
    def bondad(self):
        self.r = np.corrcoef(self.x, self.y)



# =============================================================================
# Señales adquiridas con el osci el primer día de medición. 
# 500 ms cada cuadradito y 2 V escala vertical.
# Barra de latón: (5 +- 1) mm de diámetro; (50.0 +- 0.1) cm de largo;
# (31.6 +- 0.1) cm desde el agarne hasta la cuchilla;  de peso. COMPLETAR
# =============================================================================
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (5, 5))
ax = ax.flatten()
for i in range(6):
    df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/datita{}_osci.csv'.format(str(i)))
    escala_temporal = 5e-1 # 500 ms cada cuadradito [s]
    tiempo_total = escala_temporal*10
    # Esto es para encontrar cuándo arranca y termina la señal limpia
    for k in range(len(df.datos)):
        if df.datos[k] == df.datos.max():
            comienzo = k
    k, final = comienzo, 0
    aux = df.datos[comienzo:]
    while final == 0: 
        if aux[k] == aux.min():
            final = k    
        k += 1
    # final = comienzo + 22000
    datos = df.datos[comienzo:final] #- np.mean(df.datos[comienzo:final])
    tiempo = np.linspace(0, tiempo_total, len(df.datos))[comienzo:final]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    ax[i].plot(tiempo, datos)
fig.show()

# =============================================================================
# Agarro la j-ésima (que mejor se vé)
# =============================================================================
j = 3
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/datita{}_osci.csv'.format(str(j)))
escala_temporal = 5e-1 # 500 ms cada cuadradito [s]
tiempo_total = escala_temporal*10
for k in range(len(df.datos)):
    if df.datos[k] == df.datos.max():
        comienzo = k
k, final = comienzo, 0
aux = df.datos[comienzo:]
while final == 0: 

    if aux[k] == aux.min():
        final = k    
    k += 1
datos = df.datos[comienzo:final] #- np.mean(df.datos[comienzo:final])
tiempo = np.linspace(0, tiempo_total, len(df.datos))[comienzo:final]
tstep = (tiempo.max()-tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]
fig = plt.figure('Señal osciloscopio '+ str(j) + ' día 1 de medición')
plt.plot(tiempo, datos)
fig.show()

# Tiro la fft y adquiero los picos:
picos, altura = tirafft(datos, fsamp, log = True, labels = True, picos = True,
threshold = None,
 prominence = (.5e-2, 50),
 height = None,
 distance = None,
 width = None,
 rel_height = None)

# Quedó buenisima, mirar los multiplos de la fundamental en el gráfico de arriba
for i in range(12):
    picos[0]*i 

# =============================================================================
# De esta señal, me quedo con los picos positivos para calcularle el logaritmo 
# y conseguir el coef. de decaimiento. Para esto voy a usar la función que 
# 'peaks':
# =============================================================================
picos_t, amplitud_t = peaks(tiempo = tiempo, señal = datos.tolist(), picos = True, labels = False)
picos_t, amplitud_t = np.array(picos_t), np.array(amplitud_t)

# =============================================================================
# Ahora vamos a hacer un ajuste sobre los datos en escala logarítma. Esto surge
# de asumir que el decaimiento es exponencial.
# =============================================================================
# Creo el objeto para hacer ajustes y fiteo:
reg = regresion_lineal()
reg.fit(picos_t, np.log(amplitud_t), ordenada = True)

# La matriz cov es la matriz de covarianza de los coeficientes del ajuste:
ordenada, pendiente, cov = reg.parametros[0], reg.parametros[1], reg.cov_parametros
v11, v12, v21, v22 = cov[0][0], cov[1][0], cov[0][1], cov[1][1] 

# Auxiliares par graficar:
x = np.linspace(0.5, 4.5, 10000)
ajuste = ordenada + pendiente*x 

# Esto está en el tp 3 de MEFE. Sale de calcular la covarianza para y usando los datos del ajuste:
franja_error = np.sqrt(v11 + v22*x**2 + 2*v12*x)

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    ax.plot(x, ajuste,  color = 'red', label = 'El ajuste')
    ax.plot(x, ajuste + franja_error,
               '-.', color = 'green', 
               label = 'Error del ajuste')
    ax.plot(x, ajuste - franja_error,
               '-.', color = 'green')
    ax.fill_between(x, ajuste - franja_error,
                       ajuste + franja_error, 
                       facecolor = "gray", alpha = 0.5)
    ax.scatter(picos_t, np.log(amplitud_t), marker = '.', color = 'k')
#   plt.errorbar(picos_t, np.log(amplitud_t), marker = '.', yerr = None, fmt = 'none', capsize = 5, color = 'black')
    ax.set_xlabel('Tiempo [s]', fontsize = 15)
    ax.set_ylabel('Tensión (en escala logarítmica) [log(V)]', fontsize = 15)
    ax.legend(fontsize = 15, loc = (0,0.1))
# fig.tight_layout()
fig.show()

reg.bondad()
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')
#-0.981 osea que están anticorrelacionados linealmente bastante bien

################################### REPITO EL PROCESO CON MEDICIONES DAQ DIA 2 #########################

# =============================================================================
# Señales adquiridas con el osci el segundo día de medición. 
# Barra de latón: (5 +- 1) mm de diámetro; (50.0 +- 0.1) cm de largo;
# COMPLETAR cm desde el agarre hasta la cuchilla; COMPLETAR de peso. 
# =============================================================================
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (10, 5))
ax = ax.flatten()
nombres = np.arange(5).tolist() + ['ruido']
for i in nombres:
    print(i)
    df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_daq{}.csv'.format(str(i)))
    # Esto es para encontrar cuándo arranca y termina la señal limpia
    if i == 'ruido':
        comienzo = 0
        i = 5
    else:
        comienzo = np.where(df.tension == np.amax(df.tension))[0][0]    
    datos = df.tension[comienzo:] #- np.mean(df.datos[comienzo:final])
    tiempo = tiempo_total = df.tiempo[comienzo:]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    ax[i].plot(tiempo, datos)
fig.show()

# =============================================================================
# Agarro la j-ésima (que mejor se vé)
# =============================================================================
j = 2 #'ruido'
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_daq{}.csv'.format(str(j)))
if j == 'ruido':
    comienzo = 0
    j = 5
else:
    comienzo = np.where(df.tension == np.amax(df.tension))[0][0]    
datos = df.tension[comienzo:] #- np.mean(df.datos[comienzo:final])
tiempo = tiempo_total = df.tiempo[comienzo:]
tstep = (tiempo.max()-tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]
fig = plt.figure('Señal daq '+ str(j) + ' día 2 de medición')
plt.plot(tiempo, datos)
fig.show()

# Tiro la fft y adquiero los picos:
picos, altura = tirafft(datos, fsamp, log = True, labels = True, picos = True,
threshold = None,
 prominence = (.9e-3, 50),
 height = None,
 distance = None,
 width = None,
 rel_height = None)

# Estas señales del daq no quedaron tan joyas porque tienen mucho ruido 
for i in range(12):
    picos[0]*i 

# =============================================================================
# De esta señal, me quedo con los picos positivos para calcularle el logaritmo 
# y conseguir el coef. de decaimiento. Para esto voy a usar la función que 
# 'peaks'. No funciona del todo bien porque al haber interferencia hay muchos
# picos relativos (habría que ajustar los parámetros de find_peaks):
# =============================================================================
picos_t, amplitud_t = peaks(tiempo = tiempo, señal = datos.tolist(), picos = True, labels = False)
picos_t, amplitud_t = np.array(picos_t), np.df.array(amplitud_t)

# =============================================================================
# No hago el ajuste porque no encontré los picos arriba
# =============================================================================





################################### REPITO EL PROCESO CON MEDICIONES OSCI DIA 2 #########################

# =============================================================================
# Señales adquiridas con el osci el primer día de medición. 
# 500 ms cada cuadradito y 100 mV escala vertical
# Barra de latón: (5 +- 1) mm de diámetro; (50.0 +- 0.1) cm de largo;
# (38.0 +- .2) cm desde el agarne hasta la cuchilla;  (88.85 +- 0.01)g de peso. 
# =============================================================================
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (5, 5))
ax = ax.flatten()
for i in range(5):
    df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_osci{}.csv'.format(str(i)))
    # Esto es para encontrar cuándo arranca y termina la señal limpia
    comienzo = np.where(df.tension == np.amax(df.tension))[0][0]  
    datos = df.tension[comienzo:]
    tiempo = df.tiempo[comienzo:]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    ax[i].plot(tiempo, datos)
fig.show()

# =============================================================================
# Agarro la j-ésima porque me pa que aparece el segundo modo. A CHEQUEAR
# =============================================================================
j = 1
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_osci{}.csv'.format(str(j)))
# Esto es para encontrar cuándo arranca y termina la señal limpia
comienzo = np.where(df.tension == np.amax(df.tension))[0][0]  
datos = df.tension[comienzo:]
tiempo = df.tiempo[comienzo:]
# # Primer tercio:
# aux = int(len(df.tension[comienzo:])/3)
# datos = df.tension[comienzo:comienzo + aux]
# tiempo = df.tiempo[comienzo:comienzo + aux]

# # Segundo tercio:
# aux = int(len(df.tension[comienzo:])/3)
# datos = df.tension[comienzo + aux: comienzo + 2*aux]
# tiempo = df.tiempo[comienzo + aux: comienzo + 2*aux]

# # Ultimo tercio:
# aux = int(len(df.tension[comienzo:])/3)
# datos = df.tension[comienzo + 2*aux:]
# tiempo = df.tiempo[comienzo + 2*aux:]

tstep = (tiempo.max()-tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]
fig = plt.figure('Señal osciloscopio '+ str(j) + ' día 2 de medición')
plt.plot(tiempo, datos)
fig.show()

# Tiro la fft y adquiero los picos:
picos, altura = tirafft(datos, fsamp, log = True, labels = True, picos = True,
threshold = None,
 prominence = (.9e-3, 50),
 height = None,
 distance = None,
 width = None,
 rel_height = None)

# =============================================================================
# De esta señal, me quedo con los picos positivos para calcularle el logaritmo 
# y conseguir el coef. de decaimiento. Para esto voy a usar la función que 
# 'peaks'. No me estaría saliendo xd.
# =============================================================================
tiempo_aux = [t for t in tiempo[300:]]
datos_aux = [d for d in datos[300:]]
picos_t, amplitud_t = peaks(tiempo = tiempo_aux, señal = datos_aux, picos = True,
 threshold = None,
 prominence = .1,
 height = None,
 distance = None,
 width = None,
 rel_height = None
 )
picos_t, amplitud_t = np.array(picos_t), np.array(amplitud_t)

# =============================================================================
# Ahora vamos a hacer un ajuste sobre los datos en escala logarítma. Esto surge
# de asumir que el decaimiento es exponencial.
# =============================================================================
# Creo el objeto para hacer ajustes y fiteo:
reg = regresion_lineal()
reg.fit(picos_t, np.log(amplitud_t), ordenada = True)

# La matriz cov es la matriz de covarianza de los coeficientes del ajuste:
ordenada, pendiente, cov = reg.parametros[0], reg.parametros[1], reg.cov_parametros
v11, v12, v21, v22 = cov[0][0], cov[1][0], cov[0][1], cov[1][1] 

# Auxiliares par graficar:
x = np.linspace(0.5, 4.5, 10000)
ajuste = ordenada + pendiente*x 

# Esto está en el tp 3 de MEFE. Sale de calcular la covarianza para y usando los datos del ajuste:
franja_error = np.sqrt(v11 + v22*x**2 + 2*v12*x)

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    ax.plot(x, ajuste,  color = 'red', label = 'El ajuste')
    ax.plot(x, ajuste + franja_error,
               '-.', color = 'green', 
               label = 'Error del ajuste')
    ax.plot(x, ajuste - franja_error,
               '-.', color = 'green')
    ax.fill_between(x, ajuste - franja_error,
                       ajuste + franja_error, 
                       facecolor = "gray", alpha = 0.5)
    ax.scatter(picos_t, np.log(amplitud_t), marker = '.', color = 'k')
#   plt.errorbar(picos_t, np.log(amplitud_t), marker = '.', yerr = None, fmt = 'none', capsize = 5, color = 'black')
    ax.set_xlabel('Tiempo [s]', fontsize = 15)
    ax.set_ylabel('Tensión (en escala logarítmica) [log(V)]', fontsize = 15)
    ax.legend(fontsize = 15, loc = (0,0.1))
# fig.tight_layout()
fig.show()

reg.bondad()
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')
#-0.981 osea que están anticorrelacionados linealmente bastante bien

# =============================================================================
# Gráficos para el informe de la señal y la transformada. Estoy usando el osci
# =============================================================================
j = 3
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/datita{}_osci.csv'.format(str(j)))
escala_temporal = 5e-1 # 500 ms cada cuadradito [s]
tiempo_total = escala_temporal*10
for k in range(len(df.datos)):
    if df.datos[k] == df.datos.max():
        comienzo = k
k, final = comienzo, 0
aux = df.datos[comienzo:]
while final == 0: 

    if aux[k] == aux.min():
        final = k    
    k += 1
datos = df.datos[comienzo:final] #- np.mean(df.datos[comienzo:final])
tiempo = np.linspace(0, tiempo_total, len(df.datos))[comienzo:final]
tstep = (tiempo.max()-tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]

#GRAFICOS

# Señal:
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,5))
    ax.plot(tiempo, datos, label = 'Señal')
    ax.set_xlabel('Tiempo [s]', fontsize = 12)
    ax.set_ylabel('Tensión [V]', fontsize = 12)
    ax.legend(fontsize = 12, loc = 'best')
# fig.tight_layout()
fig.show()
# Transformada:
señal_fft, N = np.fft.fft(datos), len(datos)
xf = np.linspace(0, fsamp/2, int(N/2))
yf = 2*np.abs(señal_fft[:N//2])/N

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    ax.plot(xf, yf, label = r'Transformada de Fourier de {}'.format('la señal'))
    ax.set_ylabel('Amplitud espectral (escala logarítmica)')
    ax.set_yscale('log')
    picos_x, var_inservible = find_peaks(yf, prominence = (.5e-2, 50))
    ax.set_xlim(0, xf[picos_x[-1]]+10)
    i = 0
    for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
        i += 1
        if i < 8:
            ax.plot(x_p, y_p, marker = "o", markersize = 5,
            label = '({}, {})'.format(np.round(x_p, 2), np.round(y_p, 3)))
        else:
            ax.plot(x_p, y_p, marker = "o", markersize = 5)
    ax.set_xlabel('Frecuencia [Hz]')
    ax.legend(fontsize = 12, loc = 'best')
    # fig.tight_layout()
fig.show()
I = (np.pi*(5/1000)**4)/64
masa = 88.85/1000
longitud = .38
# longitud = .316
densidad_lineal = masa/longitud
k_1 = 4.934484391
# k_1 = 5.93387364
f_1 = picos[0]
modulo = ((f_1**2)*4*np.pi**2+pendiente**2)/((I/densidad_lineal)*k_1**4)


segundo_modo = (1/(2*np.pi))*np.sqrt((I*modulo*(k_2)**4)/densidad_lineal-pendiente**2)
