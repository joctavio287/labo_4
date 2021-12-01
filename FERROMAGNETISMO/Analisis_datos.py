import numpy as np, matplotlib.pyplot as plt, pandas as pd, os
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy.stats as st

global_path = 'C:/repos/labo_4/FERROMAGNETISMO/mediciones'

# La funciones auxiliares
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
# son la 12, 13, 14, 15 y 16. A las tensiones les quito el offset y les aplico un filtro 
# savgol.
# ============================================================================================

# Creo una lista con el nombre de todos los archivos en global_path
os.chdir(global_path)
cwd = os.getcwd()
files = [str(os.path.join(cwd, f)).replace('\\','/') for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]
window = 5

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
         'tension_1': savgol_filter(CH1[1, :] - CH1[1, :].mean(), window_length = window, polyorder = 0), 'tension_2': savgol_filter((CH2[1, :]- CH1[1, :].mean())-(CH2[1, :]- CH1[1, :].mean()).mean(), window_length=11, polyorder = 0)})

# ============================================================================================
# Analisis de datos

# Primero grafico todas las curvas de histeresis juntas. Después encuentro
# la remanencia, después algún que otro gráfico auxiliar; y, finalmente, un ajuste no lineal
# para encontrar la temperatura de Curie.
# ============================================================================================

# Grafico todas las curvas de histeresis para c/medición. Hago un gráfico 3-D defino primero el mapa de colores
cmap = plt.get_cmap('plasma')
cmap_values = np.linspace(0., 1., len(eval(f'temperatura_{i}')))
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]

# Hago el gráfico:

# for i in [14,15,16]:
i = 16
fig, ax = plt.subplots(nrows = 1, ncols = 1, num = f'Medición {i}', subplot_kw={'projection': '3d'}, figsize = (7,6))
# iterador = np.arange(int(len(eval(f'temperatura_{i}'))/4))
iterador = np.arange(len(eval(f'temperatura_{i}')))
for t, c in zip(iterador, colors_rgb):
    medicion = eval(f'medicion_{i}_{t}').copy()
    ax.scatter(medicion.tension_1, t, medicion.tension_2, c = c, s = 2)
    ax.set_xlabel(r'Tensión de entrada $\propto H$ [V]')
    ax.set_ylabel('Temperatura [K]')
    ax.set_zlabel(r'Tensión de salida $\propto B$ [V]')
fig.show()

# Calculo la remanencia para c/ curva de histeresis; es decir la diferencia entre el máximo y mínimo de tensión
# en el canal 2, cuando el canal 1 vale 0

remanencia = []
for t in iterador:
    # Indices cercanos a los cruces de la tension_1 con el 0
    medicion = eval(f'medicion_{i}_{t}').copy()
    indices = np.where(np.diff(np.sign(medicion.tension_1)))[0]

    # Interpolo linealmente el 0 con veinte puntos delante y veinte detrás
    minimos_t2 = []
    for indice in indices:
        f = interpolate.interp1d(medicion.tension_1[indice-20:indice+20], medicion.tiempo_1[indice-20:indice+20])
        h = interpolate.interp1d(medicion.tiempo_2, medicion.tension_2)
        minimos_t2.append(h(f(0)))
    remanencia.append((np.mean([m for m in minimos_t2 if m >= 0]) - np.mean([m for m in minimos_t2 if m < 0]))/2)

# Gráfico auxiliar par ver una temperatura en particular, con su respectiva remanencia
t = 0
fig, ax = plt.subplots(nrows = 1, ncols = 2, num = f'Medición {i}_{t}', figsize = (9,4.5))
medicion = eval(f'medicion_{i}_{t}').copy()
ax[0].scatter(medicion.tension_1, medicion.tension_2, s = 2, label = 'Curva de histéresis', color = 'black')
# ax[0].vlines(0, ymin = 0, ymax = remanencia[t], color = 'transparent', linestyle ='--', linewidth = 2, label = r'$\propto B$ remanente')
ax[0].set_yticks([-1.0,-.5,0,.5,remanencia[t],1.0])
ax[0].set_yticklabels([-1.0,-.5,0,.5,r'$\propto B_r$',1.0])
ax[0].set_xlabel(r'Tensión de entrada $\propto H$ [V]', fontsize = 11)
ax[0].set_ylabel(r'Tensión de salida $\propto B$ [V]', fontsize = 11)
a = ax[0].get_ygridlines()
b = a[4]
b.set_color('red')
b.set_linewidth(1)
b.set_linestyle('--')
ax[0].legend(fontsize = 11)
ax[0].grid()

ax[1].scatter(medicion.tiempo_1, medicion.tension_1, s = 2, label = 'CH_1')
ax[1].scatter(medicion.tiempo_2, medicion.tension_2, s = 2, label = 'CH_2')
ax[1].set_ylabel('Tensión [V]', fontsize = 11)
ax[1].set_xlabel('Tiempo [s]', fontsize = 11)
ax[1].grid()
ax[1].legend(fontsize = 11)
fig.show()

# Hago el ajuste no lineal. Primero defino la función de ajuste
def ajuste(t, t_0, a, g, c):
    return np.piecewise(t, [t < t_0, t >= t_0], [lambda t: a*np.abs(t-t_0)**(g) + c, c])
    # return np.piecewise(t, [t < t_0, t >= t_0], [lambda t: a*(t-t_0)**(g) + c, c])

# Errores en las esclas de tensión acorde a c/ medicion:
errores = {'medicion_12_c1':8*.5/256, 'medicion_13_c1':8*.5/256, 'medicion_14_c1':8/256, 'medicion_15_c1':8*2/256, 'medicion_16_c1':8*2/256,'medicion_12_c2':8*.2/256, 'medicion_13_c2':8*.2/256, 'medicion_14_c2':8*.2/256, 'medicion_15_c2':8*.5/256, 'medicion_16_c2':8*.5/256}

# Estoy haciendo la resta entre dos valores del canal 2, entonces aparece el factor sqrt(2):
error = np.full(len(remanencia), np.sqrt(2)*errores[f'medicion_{i}_c2'])
y = None # son los datos que excluyo al final

# Initial guess
p_0 = [258,.05,.5,.04]

# Hago el ajuste
popt, pcov = curve_fit(ajuste, eval(f'temperatura_{i}'), remanencia, sigma = error, p0 = p_0)

# Armo la franja de error del ajuste
t_0, a, g, c = tuple(popt)
dt_0, da, dg, dc = tuple(np.sqrt(np.diag(pcov)))
franja_1 = lambda t: np.sqrt(
    (dt_0 * ((a*g*np.abs(t-t_0)**g)/(t-t_0)))**2+
    (da * (np.abs(t-t_0)**g))**2+
    (dg * (a*np.log(np.abs(t-t_0))*np.abs(t-t_0)**g))**2+
    (dc * (1))**2+
    2* (pcov[0][1]) * ((a*g*np.abs(t-t_0)**g)/(t-t_0)) * (np.abs(t-t_0)**g)+
    2* (pcov[0][2]) * ((a*g*np.abs(t-t_0)**g)/(t-t_0)) * (a*np.log(np.abs(t-t_0))*np.abs(t-t_0)**g)+
    2* (pcov[0][3]) * ((a*g*np.abs(t-t_0)**g)/(t-t_0)) * (1)+
    2* (pcov[1][2]) * (np.abs(t-t_0)**g) * (a*np.log(np.abs(t-t_0))*np.abs(t-t_0)**g)+
    2* (pcov[1][3]) * (np.abs(t-t_0)**g) * (1)+    
    2* (pcov[0][3]) * ((a*g*np.abs(t-t_0)**g)/(t-t_0)) * (1)+
    2* (pcov[2][3]) * (a*np.log(np.abs(t-t_0))*np.abs(t-t_0)**g) * (1)
)
# franja_1 = lambda t: np.sqrt(
#     (dt_0 * (-a*g*(t-t_0)**(g-1)))**2+
#     (da * ((t-t_0)**g))**2+
#     (dg * (a*np.log((t-t_0))*(t-t_0)**g))**2+
#     (dc * (1))**2+
#     2* (pcov[0][1]) * ((-a*g*(t-t_0)**(g-1))) * (((t-t_0)**g))+
#     2* (pcov[0][2]) * ((-a*g*(t-t_0)**(g-1))) * ((a*np.log((t-t_0))*(t-t_0)**g))+
#     2* (pcov[0][3]) * ((-a*g*(t-t_0)**(g-1))) * (1)+
#     2* (pcov[1][2]) * (((t-t_0)**g)) * ((a*np.log((t-t_0))*(t-t_0)**g))+
#     2* (pcov[1][3]) * (((t-t_0)**g)) * (1)+    
#     2* (pcov[0][3]) * ((-a*g*(t-t_0)**(g-1))) * (1)+
#     2* (pcov[2][3]) * ((a*np.log((t-t_0))*(t-t_0)**g)) * (1)
# )
x_auxiliar = np.linspace(min(eval(f'temperatura_{i}')), max(eval(f'temperatura_{i}')), 1000)

# Grafico los datos con el ajuste

fig, ax = plt.subplots(nrows = 1, ncols = 1, num = f'Ajuste-Medición {i}')
ax.scatter(eval(f'temperatura_{i}'), remanencia, s = 2, color ='black', label = 'Datos')
ax.errorbar(eval(f'temperatura_{i}'), remanencia, yerr = error, marker = '.', fmt = 'None', capsize = 1.5, color = 'black', label = 'Error de los datos')
ax.plot(x_auxiliar, ajuste(x_auxiliar, *popt), 'r-', label = 'Ajuste')
ax.plot(x_auxiliar, ajuste(x_auxiliar, *popt) + franja_1(x_auxiliar), '--', color = 'green', label = 'Error del ajuste')
ax.plot(x_auxiliar, ajuste(x_auxiliar, *popt) - franja_1(x_auxiliar), '--', color = 'green')
ax.fill_between(x_auxiliar, ajuste(x_auxiliar, *popt) -franja_1(x_auxiliar), ajuste(x_auxiliar, *popt) + franja_1(x_auxiliar), facecolor = "gray", alpha = 0.5)
ax.set_xlabel('Temperatura [K]')
ax.set_ylabel('Magnetización remanente []')
ax.grid()
ax.legend()
fig.tight_layout()
fig.show()


print('Datos del ajuste: \n\n',
    'Temperatura de Curie: ({:.1f} ± {:.1f}) K\n\n'.format(t_0, dt_0),
    'Coeficiente de proporcionalidad: ({:.3f} ± {:.3f}) V\n\n'.format(a, da),
    'Potencia: ({:.2f} ± {:.2f})\n\n'.format(g, dg),
    'Offset: ({:.3f} ± {:.3f})'.format(c, dc))

# Computo la bondad del ajuste

def chi(y_data, y_model, sigma):
    '''
    El valor esperado para chi es len(y_data) - # de datos ajustados 
    Un chi alto podría indicar error subestimado o que y_i != f(x_i)
    Un chi bajo podría indicar error sobrestimado
    '''
    return np.sum(((y_data - y_model)/sigma)**2)

modelo = ajuste(eval(f'temperatura_{i}'), t_0, a, g, c)
estadistico = chi(remanencia, modelo, error) # 76 [63.67, 88.32] debería ser len(remanencia) - 4 ± np.sqrt(2 * (len(remanencia)-4))

def significancia(y_data, parameters_adjusted, desviaciones = 1):
    nu = len(y_data) - parameters_adjusted 
    # elijo el tc como esperanza + desviaciones*sigma
    t_c_mas = len(y_data) - parameters_adjusted + desviaciones*np.sqrt(2*(len(y_data)-parameters_adjusted))
    t_c_menos = len(y_data) - parameters_adjusted - desviaciones*np.sqrt(2*(len(y_data)-parameters_adjusted))
    porcion_derecha = 1 - st.chi2.cdf(t_c_mas, nu)
    porcion_izquierda = st.chi2.cdf(t_c_menos, nu)
    return porcion_izquierda + porcion_derecha

# Probabilidad de rechazar H_0 siendo válida
alpha = significancia(remanencia, len(popt), desviaciones = .5)

def p_value(estadistico, y_data, parameters_adjusted):
    '''
    valor para contrastar con signficancia
    '''
    nu = len(y_data) - parameters_adjusted
    delta = np.abs(nu-estadistico)
    if estadistico > nu:
        return 1 - st.chi2.cdf(estadistico, nu) + st.chi2.cdf(estadistico - delta, nu)
    else:
        return st.chi2.cdf(estadistico, nu) + 1 - st.chi2.cdf(estadistico + delta, nu)

p_v_ideal = p_value(len(remanencia) - len(popt), remanencia, len(popt))
p_v = p_value(estadistico, remanencia, len(popt))

def r_squared(y_data, y_model):
    ss_res = np.sum((y_data - y_model)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1 - (ss_res / ss_tot)

r_squared(remanencia, modelo)

if p_v > alpha:
    print('Acepto la hipótesis nula')
else:
    print('Rechazo la hipótesis nula con un CL de {:.1f}%'.format((1-alpha)*100))

print('El valor esperado para el estadístico es {:.1f} ± {:.1f}'.format(len(remanencia)-len(popt), 
np.sqrt(2*(len(remanencia)-len(popt)))),
    '\nEl valor obtenido fue {:.1f}'.format(estadistico),
    '\nFijamos el estadístico crítico en {:.1f}'.format(len(remanencia)-len(popt)+np.sqrt(2*(len(remanencia)-len(popt)))),
    '\nEn base a esto determinamos el p-valor en {:.4f}'.format(p_v),
    'sobre una significancia de {:.4f}'.format(alpha))


# ===========================================================================================================
# Hipotesis nula: los datos son independientes y siguen una distribución normal.

# Elijo el estadístico T = sum_i^n [((y_i - f(x_i))/sigma_i)^2] ~ Chi^2 con n grados de libertad.
# La esperanza de este estadístico es n y la desviación es sqrt(2*n). Entonces espero que el 68 %
# de las veces T = n ± sqrt(2*n). Se define el T_critico como el valor a partir del cual rechazo 
# la hipótesis nula.

# Sin embargo, esto funciona para los datos si conozco todos los parametros que determinan el mo_
# delo. Por cada parámetro que determine por el ajuste deberé restar un grado de libertad. Enton_
# ces T = (n-m) ± sqrt(2*(n-m)), donde m es 4 en este caso.

# Análogamente se puede elije el valor de significancia 'alpha' que es P(T > t_critico | H_0) = al_
# pha; es decir, la probabilidad de rechazar H_0 siendo verdadera.

# Si T_medido es menor que T_critico se dice que acepto H_0 con un nivel de significancia alpha (o 
# bien que rechazo con un confidence level CL: 1-alpha).

# Se define el p-valor como int^{inf}_{T_m} [f_T (t') dt'], donde f_T es la distribución del esta_
# dístico. Esto es la probabilidad de obtener un resultado tanto o más incompatible con H_0 que T_m
# (i.e: P(T > T_m | H_0))

# Por lo tanto,

# p-v > alpha [P(T > T_m | H_0) > P(T > t_critico | H_0)] --->    ACEPTO     <--->      T_c < T_m

# p-v < alpha [P(T > T_m | H_0) < P(T > t_critico | H_0)] --->    RECHAZO    <--->      T_c > T_m



# ============================================================================================
# Graficos auxiliares
# ============================================================================================

# Este gráfico es para entender cómo se arma la figura de histeresis:

T = 500
cmap = plt.get_cmap('rainbow')
cmap_values = np.linspace(0., 1., int(T/20))
colors = cmap(cmap_values)
colors_rgb = ['#{0:02x}{1:02x}{2:02x}'.format(int(255*a), int(255*b), int(255*c)) for a, b, c, _ in colors]
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize= (10,5))
aux = 0
for t, c in zip(np.arange(0,T,20), colors_rgb):
    aux = t + 20 
    medicion = eval(f'medicion_{i}_{5}').copy()
    ax[1].scatter(medicion.tension_1[t:aux], medicion.tension_2[t:aux], s=2, c= c)#label = str(t), c = c)
    # ax[1].vlines(x=0,ymax=1,ymin = -1, color = 'red')
    ax[1].legend()
    ax[0].scatter(medicion.tiempo_1[t:aux], medicion.tension_1[t:aux], s = 2, c=c)
    ax[0].scatter(medicion.tiempo_2[t:aux], medicion.tension_2[t:aux], s = 2, c=c)    
fig.tight_layout()
fig.show()



# # Grafico para ver curvas de histeresis de otra manera
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import matplotlib
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


# cbaxes = inset_axes(plt.gca(), width="2%", height="60%", loc=4)
# cbar = matplotlib.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm, ticks=ticks)
# cbar.set_label('Temperatura')
# cbar.ax.set_yticklabels(ticks, fontsize=12)
# ax.legend()
# fig.show()


# H = B/mu_0 - M -----> H=0 --> mu_0*M = B_s 

# Tension primario proporcional a H por ampere