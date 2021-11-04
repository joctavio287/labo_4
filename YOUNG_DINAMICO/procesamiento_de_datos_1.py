import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd, os
# Es el path al directorio contenedor de ffts.py
path = "C:/repos/labo_4/YOUNG_DINAMICO/"
os.chdir(path)
from ffts import *
path = "C:/repos/labo_4/"
os.chdir(path)
from  funciones import *

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
# Análisis del ruido
# =============================================================================
j = 'ruido'
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



################################### REPITO EL PROCESO CON MEDICIONES OSCI DIA 2 #########################

# =============================================================================
# Señales adquiridas con el osci el primer día de medición. 
# 500 ms cada cuadradito y 200 mV escala vertical
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
# Agarro la j-ésima porque me pa que aparece el segundo modo. 
# =============================================================================

############################## APARTIR DE ACÁ SE ENCUENTRA LA INFORMACIÓN QUE UTILIZAMOS EN EL INFORME ###########################################

j = 1
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_osci{}.csv'.format(str(j)))
# Esto es para encontrar cuándo arranca y termina la señal limpia
comienzo = np.where(df.tension == np.amax(df.tension))[0][0]  
datos = df.tension[comienzo:]
datos.max()-datos.min()
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
# 'peaks'. No funciona del todo bien porque al haber interferencia hay muchos
# picos relativos (habría que ajustar los parámetros de find_peaks):
# =============================================================================

# esto es pq' hay ruido de la interferencia con el segundo modo en los primeros datos
tiempo_aux = [t for t in tiempo[300:]]
datos_aux = [d for d in datos[300:]]

picos_t, amplitud_t = peaks(tiempo = tiempo_aux, señal = datos_aux, labels = False,
 picos = True,
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
error = 0.00625*1.5 # 1.6V/256 + %50
cov_error = np.identity(len(amplitud_t))*error**2
reg = regresion_lineal(picos_t, np.log(amplitud_t), cov_y = cov_error, n = 1, ordenada = True)
reg.fit()

# La matriz cov es la matriz de covarianza de los coeficientes del ajuste:
ordenada, pendiente, cov = reg.parametros[0], reg.parametros[1], reg.cov_parametros
v11, v12, v21, v22 = cov[0][0], cov[1][0], cov[0][1], cov[1][1] 

# Auxiliares par graficar:
x = np.linspace(0.5, 4.5, 10000)
ajuste = ordenada + pendiente*x 

# Esto está en el tp 3 de MEFE. Sale de calcular la covarianza para y usando los datos del ajuste:
franja_error = np.sqrt(v11 + v22*x**2 + 2*v12*x)

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 7))
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
    ax.errorbar(picos_t, np.log(amplitud_t), marker = '.', yerr = error, fmt = 'none', capsize = 2, color = 'black')
    ax.set_xlabel('Tiempo [s]', fontsize = 13)
    ax.set_ylabel('Tensión (en escala logarítmica) [log(V)]', fontsize = 13)
    ax.legend(fontsize = 11, loc = (0,0.1))
fig.tight_layout()
fig.show()

# Calculo el coeficiente de correlación lineal
reg.bondad()
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')

# =============================================================================
# CALCULO DE MODULO DE YOUNG Y FRECUENCIA DEL SEGUNDO MODO. ADEMÁS, SUS 
# RESPECTIVOS ERRORES.
# =============================================================================
# =============================================================================
# Propagación de errores del módulo de Young
# =============================================================================
# EN PASCALES = [kg/m*s^2]

f_1, k_1, k_2, a, d, m, l,  = picos[0].copy(), 4.934484391, 12.3528714, pendiente.copy()[0], 5/1000, 88.85/1000, .38
longitud_entera = .5
dic_young = {'variables': [('f_1', f_1, 0), ('k_1',k_1,0), ('a',pendiente,float(np.sqrt(v22))), ('d',d,.05/1000), ('m',m,.01/1000), ('l',l,1/1000)],
    'expr': ('E', '((f_1**2)*4*np.pi**2+a**2)/((((np.pi*(d)**4)/64)/(m/l))*k_1**4)')}
propaga = propagacion_errores(dic_young)
propaga.fit()
# EN GIGAPASCALES
E, errorE = propaga.valor, propaga.error
print(E/1e9, errorE/1e9)

# =============================================================================
# Propagación de errores del segundo modo
# =============================================================================
f_1, k_1, k_2, a, d, m, l  = picos[0].copy(), 4.934484391, 12.3528714, pendiente.copy()[0], 5/1000, 88.85/1000, .38
E, errorE = propaga.valor[0], propaga.error[0]

dic_modo_2 = {'variables': [('E', E, errorE), ('k_2', k_2, 0), ('a', a, 0.000960), ('d', d, 1/1000), ('m', m, .01/1000), ('l', l, 1/1000)],
 'expr': ('m_2', '(1/(2*np.pi))*((((np.pi*(d)**4)/64)*E*(k_2)**4)/(m/l)-a**2)**(1/2)')}

propaga2 = propagacion_errores(dic_modo_2)
propaga2.fit()
modo2, errormodo2 = propaga2.valor, propaga2.error
print(modo2, errormodo2)
