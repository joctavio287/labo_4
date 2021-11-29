from scipy.optimize import curve_fit
import itertools
import time, pyvisa, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as mlt, pandas as pd, os, matplotlib
from scipy import interpolate
from scipy.signal import filter_design, savgol_filter
global_path = 'C:/repos/labo_4/FERROMAGNETISMO/mediciones'

# Definición de funciones auxiliares
def funcion_integradora(x, y, offset = True):
    '''
    Realiza la integral numerica de un set de datos (x, y)
    INPUT: 
    -x--> np.array: el tiempo de la señal (de no tener inventarlo convenientemente)
    -y--> np.array: la señal.
    -offset--> Bool: si la señal está corrida del cero, el offset genera errores. 
    '''
    if offset:
        y -= y.mean()
    T = x.max() - x.min()
    return np.cumsum(y) * (T/len(x))  # HAY UN OFFSET SOLUCIONAR
def funcion_conversora_temp(t):
    '''
    Conversor para Pt100 platinum resistor (http://www.madur.com/pdf/tools/en/Pt100_en.pdf)
    INPUT:
    t --> np.array: temperatura[Grados Celsius]. 
    OUTPUT:
    r --> np.array: resistencia[Ohms].
     '''
    R_0 = 100 # Ohms; resistencia a 0 grados Celsius
    A = 3.9083e-3 # grados a la menos 1
    B = -5.775e-7 # grados a la menos 2
    C = -4.183e-12 # grados a la menos 4
    return np.piecewise(t, [t < 0, t >= 0], [lambda t: R_0*(1+A*t+B*t**2+C*(t-100)*t**3), lambda t: R_0*(1+A*t+B*t**2)])

# Escala auxiliar para hacer la transformación a temperatura
temperaturas_auxiliar = np.linspace(-300, 300, 100000) 
conversor = {r: t for r, t in zip(funcion_conversora_temp(temperaturas_auxiliar), temperaturas_auxiliar)}

# ============================================================================================
# Leo los datos, determiné viendo plots de todos los datos que las únicas mediciones buenas 
# son la 12, 13, 14, 15 y 16.
# ============================================================================================

# Creo una lista con el nombre de todos los archivos en global_path
os.chdir(global_path)
cwd = os.getcwd()
files = [str(os.path.join(cwd, f)).replace('\\','/') for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

# Armo dataframes para cada una de las temperaturas, sobre todas las mediciones
for i in [12, 13, 14, 15, 16]:
    resistencias = np.loadtxt([f for f in files if f'Medicion {i}' in f and 'Resistencias.txt' in f][0], delimiter = ',', dtype = float)
    tiempo_t = resistencias[1, :]
    temperatura = [conversor[min(conversor.keys(), key = lambda x:abs(x-r))]  + 273.15 for r in resistencias[0,:]] # + 273.15
    locals()[f'temperatura_{i}'] = temperatura
    locals()[f'tiempo_t_{i}'] = tiempo_t
    for j in range(1, len(temperatura) + 1):
        CH1 = np.loadtxt(global_path + f'/Medicion {i} - CH1 - Resistencia {j}.0.txt', delimiter = ',', dtype = float)
        CH2 = np.loadtxt(global_path + f'/Medicion {i} - CH2 - Resistencia {j}.0.txt', delimiter = ',', dtype = float)
        # cuando escribo los datos corrigo offsets, primero respecto al CH1 y en base a eso el CH2 y agrego filtro sav
        locals()[f'medicion_{i}_{j-1}'] = pd.DataFrame(data = {'tiempo_t': tiempo_t[j-1], 'temperatura': temperatura[j-1], 'tiempo_1' : CH1[0,:], 'tiempo_2' : CH2[0,:],
         'tension_1': savgol_filter(CH1[1, :] - CH1[1, :].mean(), window_length=11, polyorder = 0), 'tension_2': savgol_filter((CH2[1, :]- CH1[1, :].mean())-(CH2[1, :]- CH1[1, :].mean()).mean(), window_length=11, polyorder = 0)})
        
# eval(f'medicion_{i}_{j-1}.tension_1')


#########################################

# Grafico de las curvas de histeresis
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., len(eval(f'temperatura_{i}')))
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]
# for i in [12,13,14,15,16]:
i = 14
fig, ax = plt.subplots(nrows = 1, ncols = 1, num = f'Medición {i}', subplot_kw={'projection': '3d'})
temps = eval(f'temperatura_{i}')
for t, c in zip(list(range(int(len(eval(f'temperatura_{i}'))/1))), colors_rgb):
# for t, c in zip(list(range(60,80)), colors_rgb):
    medicion = eval(f'medicion_{i}_{t}').copy()
    ax.scatter(medicion.tension_1, t, medicion.tension_2, c = c, s = 2)
    ax.set_xlabel('Tensión de entrada [V]')
    ax.set_ylabel('Temperatura [K]')
    ax.set_zlabel('Tensión de salida [V]')
fig.show()
for t in [45,47,48,49,50,51,52,53,55]:
    fig, ax = plt.subplots(nrows = 1, ncols = 2, num = f'Medición {i}_{t}')
    medicion = eval(f'medicion_{i}_{t}').copy()
    ax[0].scatter(medicion.tension_1, medicion.tension_2,s = 2)
    ax[0].vlines(0, ymin = -remanencia[t], ymax = remanencia[t], color = 'red')
    ax[0].set_xlabel('Tensión de entrada [V]')
    ax[0].set_ylabel('Tensión de salida [V]')
    ax[1].scatter(medicion.tiempo_1, medicion.tension_1, s = 2)
    ax[1].scatter(medicion.tiempo_2, medicion.tension_2, s = 2)
    fig.show()



i = 15
remanencia = []
for t in list(range(int(len(eval(f'temperatura_{i}'))/1))):
    medicion = eval(f'medicion_{i}_{t}').copy()
    # Encuentro cruces con tension_1 = 0, que se coinciden en el gráfico de histeresis con la remanente
    # Indices cercanos
    indices = np.where(np.diff(np.sign(medicion.tension_1)))[0]

    # Interpolo linealmente con veinte puntos delante y veinte detrás
    tiempos_min = []
    minimos_t2 = []
    for indice in indices:
        f = interpolate.interp1d(medicion.tension_1[indice-20:indice+20], medicion.tiempo_1[indice-20:indice+20])
        tiempo = f(0)
        tiempos_min.append(tiempo)
        h = interpolate.interp1d(medicion.tiempo_2, medicion.tension_2)
        minimos_t2.append(h(tiempo))
    aux_1, aux_2 = [], []
    for minimo in minimos_t2:
        if minimo < 0:
            aux_1.append(minimo)
        else:
            aux_2.append(minimo)
    tension_2_remanente_min, tension_2_remanente_max = np.mean(aux_1), np.mean(aux_2)
    remanencia.append((tension_2_remanente_max-tension_2_remanente_min)/2)

y = 20

def func(t, t_0, a, g, c):
    # return a*(t_0-t)**(g) 
    return np.piecewise(t, [t < t_0, t >= t_0], [lambda t: a*np.abs(t_0-t)**(g) + c, c])
# def func(t, t_0)    
#     return A + B|T − TC|−α

# Errores en las esclas de tensión acorde a c/ medicion:
errores = {'medicion_12_c1':8*.5/256, 'medicion_13_c1':8*.5/256, 'medicion_14_c1':8/256, 'medicion_15_c1':8*2/256, 'medicion_16_c1':8*2/256,
'medicion_12_c2':8*.2/256, 'medicion_13_c2':8*.2/256, 'medicion_14_c2':8*.2/256, 'medicion_15_c2':8*.5/256, 'medicion_16_c2':8*.5/256}

# Estoy haciendo la resta entre dos valores del canal 2:
error = np.full(len(remanencia), np.sqrt(2)*errores[f'medicion_{i}_c2'])



popt, pcov = curve_fit(func, eval(f'temperatura_{i}')[:-y], remanencia[:-y], sigma = error[:-y], p0 = [258,.05,.5,.04])#, bounds = (0, [270, 1, .6, 5]))
t_0, a, g, c = tuple(popt)
dt_0, da, dg, dc = tuple(np.sqrt(np.diag(pcov)))
franja_1 = lambda t: np.sqrt(
    (dt_0 * ((a*g*np.abs(t_0-t)**g)/(t_0-t)))**2+
    (da * (np.abs(t_0-t)**g))**2+
    (dg * (a*np.log(np.abs(t_0-t))*np.abs(t_0-t)**g))**2+
    (dc * (1))**2+
    2* (pcov[0][1]) * ((a*g*np.abs(t_0-t)**g)/(t_0-t)) * (np.abs(t_0-t)**g)+
    2* (pcov[0][2]) * ((a*g*np.abs(t_0-t)**g)/(t_0-t)) * (a*np.log(np.abs(t_0-t))*np.abs(t_0-t)**g)+
    2* (pcov[0][3]) * ((a*g*np.abs(t_0-t)**g)/(t_0-t)) * (1)+
    2* (pcov[1][2]) * (np.abs(t_0-t)**g) * (a*np.log(np.abs(t_0-t))*np.abs(t_0-t)**g)+
    2* (pcov[1][3]) * (np.abs(t_0-t)**g) * (1)+    
    2* (pcov[0][3]) * ((a*g*np.abs(t_0-t)**g)/(t_0-t)) * (1)+
    2* (pcov[2][3]) * (a*np.log(np.abs(t_0-t))*np.abs(t_0-t)**g) * (1)
)



plt.figure()
plt.scatter(eval(f'temperatura_{i}'), remanencia, s=2, color ='black')
plt.scatter(eval(f'temperatura_{i}')[40:55], remanencia[40:55], s=8, color ='red')
plt.errorbar(eval(f'temperatura_{i}'), remanencia, yerr = error, marker = '.', fmt = 'None', capsize = 1.5, color = 'black', label = 'Error de los datos')
plt.plot(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000),
 func(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), *popt), 'r-',
  label='fit: t_0=%5.3f, escala=%5.3f, potencia=%5.3f, offset =%5.3f. +-: %5.3f,%5.3f,%5.3f,%5.3f ' % tuple(popt.tolist() + np.sqrt(np.diag(pcov)).tolist()))
plt.plot(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), func(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), *popt)
+franja_1(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000)), '--', color = 'green')
plt.plot(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), func(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), *popt)
-franja_1(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000)), '--', color = 'green')
plt.fill_between(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), func(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), *popt)
-franja_1(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000)),
                       func(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000), *popt)
+franja_1(np.linspace(min(eval(f'temperatura_{i}')[:-y]), max(eval(f'temperatura_{i}')[:-y]),1000)), 
                       facecolor = "gray", alpha = 0.5)
plt.legend()
plt.show(block=False)


popt[1]-






































# # Este gráfico es para entender cómo se arma la figura de histeresis:

# T = 400
# cmap = plt.get_cmap('rainbow')
# cmap_values = np.linspace(0., 1., int(T/20))
# colors = cmap(cmap_values)
# colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]
# fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize= (10,5))
# aux = 0
# for t, c in zip(np.arange(0,T,20), colors_rgb):
#     aux = t + 20 
#     medicion = eval(f'medicion_{i}_{5}').copy()
#     ax[1].scatter(medicion.tension_1[t:aux], medicion.tension_2[t:aux], s=2, c= c)#label = str(t), c = c)
#     # ax[1].vlines(x=0,ymax=1,ymin = -1, color = 'red')
#     ax[1].legend()
#     ax[0].scatter(medicion.tiempo_1[t:aux], medicion.tension_1[t:aux], s = 2, c=c)
#     ax[0].scatter(medicion.tiempo_2[t:aux], medicion.tension_2[t:aux], s = 2, c=c)    
# fig.tight_layout()
# fig.show()



# # Grafico 2
# cmap = plt.get_cmap('plasma')
# cmap_values = np.linspace(0., 1., len(eval(f'temperatura_{i}')))
# colors = cmap(cmap_values)
# colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]
# fig, ax = plt.subplots(nrows = 1, ncols = 1, num = f'Medición {i}')

# for t, c in zip(list(range(len(eval(f'temperatura_{i}')))), colors_rgb):
#     medicion = eval(f'medicion_{i}_{t+1}').copy()
#     ax.scatter(medicion.tension_1, medicion.tension_2, c = c, s = 2)
#     ax.set_xlabel('Tensión de entrada [V]')
#     ax.set_ylabel('Tensión de salida [V]')
# norm = matplotlib.colors.Normalize(vmin = np.array(eval(f'temperatura_{i}')).min(), vmax = np.array(eval(f'temperatura_{i}')).max())
# ticks = np.arange(np.array(eval(f'temperatura_{i}')).min(), np.array(eval(f'temperatura_{i}')).max(), 50)
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# cbaxes = inset_axes(plt.gca(), width="2%", height="60%", loc=4)
# cbar = matplotlib.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm, ticks=ticks)
# cbar.set_label('Temperatura')
# cbar.ax.set_yticklabels(ticks, fontsize=12)
# ax.legend()
# fig.show()
