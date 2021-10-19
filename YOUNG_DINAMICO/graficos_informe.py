import scipy.stats as sp, numpy as np, matplotlib.pyplot as plt, pandas as pd, os
from scipy.signal import find_peaks  

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

#######################################################

j = 1
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_osci{}.csv'.format(str(j)))
# Esto es para encontrar cuándo arranca y termina la señal limpia
comienzo = np.where(df.tension == np.amax(df.tension))[0][0]  
datos = df.tension[comienzo:]
tiempo = df.tiempo[comienzo:]
tstep = (tiempo.max()-tiempo.min())/len(tiempo)
fsamp = 1/tstep # frecuencia de sampleo [HZ]

# SEÑAL:
with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 7), sharey = True)
    fig.supylabel('Tensión [V]', fontsize = 13)
    fig.supxlabel('Tiempo [s]', fontsize = 13)
    axs = axs.flatten()
    axs[0].plot(tiempo, datos, label = 'Señal entera')
    # axs[0].set_xlabel('Tiempo [s]', fontsize = 13)
    # axs[0].set_ylabel('Tensión [V]', fontsize = 13)
    axs[0].legend(fontsize = 11, loc = 'best')
    aux_1, aux_2 = 0, int(len(datos)/10)
    axs[1].plot(tiempo[aux_1:aux_2], datos[aux_1:aux_2], label = 'Primer décimo de la señal')
    # axs[1].set_xlabel('Tiempo [s]', fontsize = 13)
    # axs[1].set_ylabel('Tensión [V]', fontsize = 13)
    axs[1].legend(fontsize = 11, loc = 'best')
    aux_1, aux_2 = int(len(datos)/10), 2*int(len(datos)/10) 
    axs[2].plot(tiempo[aux_1:aux_2], datos[aux_1:aux_2], label = 'Segundo décimo de la señal')
    # axs[2].set_xlabel('Tiempo [s]', fontsize = 13)
    # axs[2].set_ylabel('Tensión [V]', fontsize = 13)
    axs[2].legend(fontsize = 11, loc = 'best')
    aux_1, aux_2 = 2*int(len(datos)/10), 3*int(len(datos)/10)
    axs[3].plot(tiempo[aux_1:aux_2], datos[aux_1:aux_2], label = 'Tercer décimo de la señal')
    # axs[3].set_xlabel('Tiempo [s]', fontsize = 13)
    # axs[3].set_ylabel('Tensión [V]', fontsize = 13)
    axs[3].legend(fontsize = 11, loc = 'best')
    
fig.subplots_adjust( 
left  = 0.09,  # the left side of the subplots of the figure
right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
bottom = 0.075,   # the bottom of the subplots of the figure
top = 0.999,      # the top of the subplots of the figure
wspace = 0.05,   # the amount of width reserved for blank space between subplots
hspace = 0.1)   # the amount of height reserved for white space between subplots
# fig.show()


fig.savefig('C:/Users/jocta/Documents/LaTex/Modulo_young/señal.jpg', dpi=1200)

# LINEAL:
j = 1
df = pd.read_csv('C:/repos/labo_4/YOUNG_DINAMICO/Mediciones/medicio_osci{}.csv'.format(str(j)))
# Esto es para encontrar cuándo arranca y termina la señal limpia
comienzo = np.where(df.tension == np.amax(df.tension))[0][0]  
datos = df.tension[comienzo:]
tiempo = df.tiempo[comienzo:]

tiempo_aux = [t for t in tiempo[300:]]
datos_aux = [d for d in datos[300:]]
picos_x, var_inservible = find_peaks(datos_aux,prominence = .1)
picos_t, amplitud_t = np.array([tiempo_aux[x] for x in picos_x]), np.array([datos_aux[x] for x in picos_x])

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
x = np.linspace(0.5, 5, 10000)
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
#   plt.errorbar(picos_t, np.log(amplitud_t), marker = '.', yerr = None, fmt = 'none', capsize = 5, color = 'black')
    ax.set_xlabel('Tiempo [s]', fontsize = 13)
    ax.set_ylabel('Tensión (en escala logarítmica) [log(V)]', fontsize = 13)
    ax.legend(fontsize = 11, loc = (0,0.1))
fig.tight_layout()
fig.show()

reg.bondad()
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')
#-0.981 osea que están anticorrelacionados linealmente bastante bien
fig.savefig('C:/Users/jocta/Documents/LaTex/Modulo_young/lineal.jpg', dpi=1200)


# TRANSFORMADA:

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 7), sharey = True)
    fig.supylabel('Amplitud espectral (escala logarítmica)', fontsize = 13)
    fig.supxlabel('Frecuencia [Hz]', fontsize = 13)
    axs = axs.flatten()
    # Transformada entera:
    datos = [d for d in df.tension[comienzo:]]
    tiempo = np.array([t for t in df.tiempo[comienzo:]])
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    señal_fft, N = np.fft.fft(datos), len(datos)
    xf = np.linspace(0, fsamp/2, int(N/2))
    yf = (2*np.abs(señal_fft[:N//2])/N)
    
    axs[0].plot(xf, yf, label = 'Señal entera')
    # axs[0].set_xlabel('Frecuencia [Hz]', fontsize = 13)
    # axs[0].set_ylabel('Amplitud espectral (escala logarítmica)', fontsize = 13)
    axs[0].set_yscale('log')
    # axs[0].set_xlim(0, 2*xf[-1]/4)
    picos_x, var_inservible = find_peaks(yf, prominence = (.5e-2, 1))
    for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
        if int(x_p) != 44:
            axs[0].plot(x_p, y_p, marker = "o", markersize = 5,
            label = 'Coordenadas del pico: ({}, {})'.format(np.round(x_p, 2), np.round(y_p, 4)))
    axs[0].legend(fontsize = 11, loc = 'best')

    # Transformada primer tramo;
    datos = [d for d in df.tension[comienzo:]]
    tiempo = np.array([t for t in df.tiempo[comienzo:]])
    aux_1, aux_2 = 0, int(len(datos)/10)
    datos = datos[comienzo + aux_1: comienzo + aux_2]
    tiempo = tiempo[comienzo + aux_1: comienzo + aux_2]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    señal_fft, N = np.fft.fft(datos), len(datos)
    xf = np.linspace(0, fsamp/2, int(N/2))
    yf = (2*np.abs(señal_fft[:N//2])/N)
    
    axs[1].plot(xf, yf, label = 'Primer décimo señal')
    # axs[1].set_xlabel('Frecuencia [Hz]', fontsize = 13)
    # axs[1].set_ylabel('Amplitud espectral (escala logarítmica)', fontsize = 13)
    axs[1].set_yscale('log')
    axs[1].set_xlim(0, 2*xf[-1]/4)
    picos_x, var_inservible = find_peaks(yf, prominence = (.5e-2, 1))
    for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
        if int(x_p) != 44:
            axs[1].plot(x_p, y_p, marker = "o", markersize = 5,
            label = 'Coordenadas del pico: ({}, {})'.format(np.round(x_p, 2), np.round(y_p, 4)))
    axs[1].legend(fontsize = 11, loc = 'best')

    # Transformada segundo tramo:
    datos = [d for d in df.tension[comienzo:]]
    tiempo = np.array([t for t in df.tiempo[comienzo:]])
    aux_1, aux_2 = int(len(datos)/10), 2*int(len(datos)/10) 
    datos = datos[comienzo + aux_1: comienzo + aux_2]
    tiempo = tiempo[comienzo + aux_1: comienzo + aux_2]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    señal_fft, N = np.fft.fft(datos), len(datos)
    xf = np.linspace(0, fsamp/2, int(N/2))
    yf = (2*np.abs(señal_fft[:N//2])/N)
    
    axs[2].plot(xf, yf, label = 'Segundo décimo la señal')
    # axs[2].set_xlabel('Frecuencia [Hz]', fontsize = 13)
    # axs[2].set_ylabel('Amplitud espectral (escala logarítmica)', fontsize = 13)
    axs[2].set_yscale('log')
    axs[2].set_xlim(0, 2*xf[-1]/4)
    picos_x, var_inservible = find_peaks(yf, prominence = (.5e-2, 1))
    for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
        if int(x_p) != 44:
            axs[2].plot(x_p, y_p, marker = "o", markersize = 5,
            label = 'Coordenadas del pico: ({}, {})'.format(np.round(x_p, 2), np.round(y_p, 4)))
    axs[2].legend(fontsize = 11, loc = 'best')

    # Transformada tercer tramo:
    datos = [d for d in df.tension[comienzo:]]
    tiempo = np.array([t for t in df.tiempo[comienzo:]])
    aux_1, aux_2 = 2*int(len(datos)/10), 3*int(len(datos)/10) 
    datos = datos[comienzo + aux_1: comienzo + aux_2]
    tiempo = tiempo[comienzo + aux_1: comienzo + aux_2]
    tstep = (tiempo.max()-tiempo.min())/len(tiempo)
    fsamp = 1/tstep # frecuencia de sampleo [HZ]
    señal_fft, N = np.fft.fft(datos), len(datos)
    xf = np.linspace(0, fsamp/2, int(N/2))
    yf = (2*np.abs(señal_fft[:N//2])/N)
    
    axs[3].plot(xf, yf, label = 'Tercer décimo de la señal')
    # axs[3].set_xlabel('Frecuencia [Hz]', fontsize = 13)
    # axs[3].set_ylabel('Amplitud espectral (escala logarítmica)', fontsize = 13)
    axs[3].set_yscale('log')
    axs[3].set_xlim(0, 2*xf[-1]/4)
    picos_x, var_inservible = find_peaks(yf, prominence = (.5e-2, 1))
    for x_p, y_p in zip([xf[x] for x in picos_x], [yf[x] for x in picos_x]):
        if int(x_p) != 44:
            axs[3].plot(x_p, y_p, marker = "o", markersize = 5,
            label = 'Coordenadas del pico: ({}, {})'.format(np.round(x_p, 2), np.round(y_p, 4)))
    axs[3].legend(fontsize = 11, loc = 'best')
# fig.tight_layout()
fig.subplots_adjust( 
left  = 0.09,  # the left side of the subplots of the figure
right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
bottom = 0.075,   # the bottom of the subplots of the figure
top = 0.999,      # the top of the subplots of the figure
wspace = 0.05,   # the amount of width reserved for blank space between subplots
hspace = 0.1)   # the amount of height reserved for white space between subplots

# fig.show()

fig.savefig('C:/Users/jocta/Documents/LaTex/Modulo_young/transformada.jpg', dpi=1200)

