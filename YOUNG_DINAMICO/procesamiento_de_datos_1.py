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
        cov_parametros = np.linalg.inv(np.dot(A.T, np.dot(inversa_cov, A)))
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
reg = regresion_lineal()
error = 0.00625*1.5 # 1.6V/256 + %50
cov_error = np.diag((np.full(amplitud_t.shape, error))**2)
reg.fit(picos_t, np.log(amplitud_t), cov_y = cov_error, ordenada = True)

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

I = (np.pi*(5/1000)**4)/64 # momento de inercia seccional
masa = 88.85/1000
longitud = .38
longitud_entera = .5
# longitud = .316 # osci día1
densidad_lineal = masa/longitud#/longitud_entera
k_1 = 4.934484391
# k_1 = 5.93387364 # osci día1
f_1 = picos[0]

# EN PASCALES = [kg/m*s^2]
modulo = ((f_1**2)*4*np.pi**2+pendiente**2)/((I/densidad_lineal)*k_1**4) 
# EN GIGAPASCALES
modulo_gpa = modulo/1e9

k_2 = 12.3528714
# k_2 = 14.85471878 # osci día1
# EN HZ
segundo_modo = (1/(2*np.pi))*np.sqrt((I*modulo*(k_2)**4)/densidad_lineal-pendiente**2)

# PROPAGACIÓN DE ERRORES:

from sympy import symbols, solve, nsolve, cos, cosh, sin, sinh, exp, lambdify, latex, diff, sqrt

def suma_cuadrada(lista):
    suma = 0
    for el in lista:
        suma += el**2
    return suma

E, f, a, r, i, k, d, m, l = symbols("E f a r i k d m l", real = True)

# Escribo la fórmula del módulo de Young en término de todas las variables medidas:
numerador = ((2*np.pi*f)**2 + a**2)*(m/l)
denominador = (np.pi*(d**4))*((4.934484391/l)**4)
E = numerador/denominador

der_a, da = diff(E, a), 0.00009
der_m, dm = diff(E, m), 1/1000
der_l, dl = diff(E, l), .01/1000 
der_d, dd = diff(E, d), 1/1000


auxiliar_E = suma_cuadrada([der_a*da, der_m*dm, der_l*dl, der_d*dd])

lam_E = lambdify([a, m, l, d, f], auxiliar_E, modules = ['numpy'])

dE = np.sqrt(lam_E(-.052710, 88.85/1000, .38, 5/1000, f_1))
dE_gpa = dE/1e9
print('El valor medido para el módulo de Young del latón es de: ({} ± {}) GPa'.format(
    np.round(modulo_gpa, 5),
    np.round(dE_gpa, 5)))

# Escribo la fórmula de la frecuencia del segundo modo en término de todas las variables medidas:

freq, d, E, m, l, a    = symbols("freq d E m l a", real = True)
freq = ((E*l*((np.pi*(d)**4)/64))/m - a**2)*1/(2*np.pi)

der_a, da = diff(freq, a), 0.00009
der_m, dm = diff(freq, m), 1/1000
der_l, dl = diff(freq, l), .01/1000 
der_d, dd = diff(freq, d), 1/1000
der_E, dE = diff(freq, E), np.sqrt(lam_E(-.052710, 88.85/1000, .38, 5/1000, f_1))

auxiliar_freq = suma_cuadrada([der_a*da, der_m*dm, der_l*dl, der_d*dd, der_E*dE])

lam_f = lambdify([a, m, l, d, E], auxiliar_freq, modules = ['numpy'])

dfreq = np.sqrt(lam_f(-.052710, 88.85/1000, .38, 5/1000, modulo))

print('El valor esperado para el segundo modo es de: ({} ± {}) GPa'.format(
    np.round(segundo_modo, 5),
    np.round(dfreq, 5)))