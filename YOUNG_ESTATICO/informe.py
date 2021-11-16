import numpy as np, matplotlib.pyplot as plt, pandas as pd
from funciones import *
import ast

masas_2 = {'base':14.6624,1:2.1160,2:2.0399,3:1.6966,4:10.046,5:2.1535,
6:5.0952,7:20.0350,10:4.7474,11:9.9163,12:2.1593,15:2.1468}
imagenes_2_grana = {
'164151598': 'fondo',
'171219177': 0,
'171340262': masas_2['base'],
'171442093': masas_2['base']+masas_2[3],
'172921941': masas_2['base']+masas_2[10],
'172631960': masas_2['base']+masas_2[11],
'171539519': masas_2['base']+masas_2[3]+masas_2[5],
'172723255': masas_2['base']+masas_2[11]+masas_2[4],
'171740093': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15],
'171839967': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7],
'171921081': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6],
'172116591': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1],
'172238962': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2],
'172421706': masas_2['base']+masas_2[3]+masas_2[5]+masas_2[15]+masas_2[7]+masas_2[6]+masas_2[1]+masas_2[2]+masas_2[12],
}
mediciones_2_grana = {key:None for key in imagenes_2_grana.keys()}
nombres_2_grana = list(imagenes_2_grana.keys())

# Gráfico calibracion:
l = 0
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
    axs[0].set_xlabel('Píxeles eje horizontal', fontsize = 12)
    axs[0].set_ylabel('Píxeles eje vertical', fontsize = 12)
    axs[0].imshow(imagen[90:390,310:430])
    imagen = imagen[:,:, 0]
    axs[1].plot(imagen[90:390,310:430].sum(axis = 1))
    # axs[1].set_xticklabels(np.arange(100,550,50))
    axs[1].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    axs[1].set_xlabel('Píxeles eje vertical', fontsize = 12)
    clicks = plt.ginput(n = -1, timeout = -1)
    for i,j  in clicks:
        plt.plot(i,j, '.', color = 'r')
    axs[1].set_yticks([])
    axs[1].legend([r'$\propto$ Intensidad lumínica', 'Mínimos relativos'], loc = 'best', fontsize = 10)
    fig.subplots_adjust( 
    left  = 0.0001,  # the left side of the subplots of the figure
    right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    bottom = 0.12,   # the bottom of the subplots of the figure
    top = 0.99,      # the top of the subplots of the figure
    wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    hspace = 0.075)   # the amount of height reserved for white space between subplots

    fig.show()
    # fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/calibracion.png', dpi=1200)


# Gráfico perfil:
l = 10
imagen = plt.imread('C:/repos/labo_4/YOUNG_ESTATICO/Mediciones_2_grana-alteradas/' + nombres_2_grana[l] + '.jpg')

with plt.style.context('seaborn-whitegrid'):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
    axs[0].set_xlabel('Píxeles eje horizontal', fontsize = 12)
    axs[0].set_ylabel('Píxeles eje vertical', fontsize = 12)
    axs[0].imshow(imagen[100:550,600:800])
    imagen = imagen[:,:, 0]
    axs[1].plot(imagen[100:550,650:770].sum(axis = 1))
    # axs[1].set_xticklabels(np.arange(100,550,50))
    axs[1].set_ylabel('Suma de píxeles sobre el eje horizontal', fontsize = 12)
    axs[1].set_xlabel('Píxeles eje vertical', fontsize = 12)
    clicks = plt.ginput(n = -1, timeout = -1)
    for i,j  in clicks:
        plt.plot(i,j, '.', color = 'r')
    axs[1].set_yticks([])
    axs[1].legend([r'$\propto$ Intensidad lumínica', 'Mínimos relativos'], loc = 'best', fontsize = 10)
    fig.subplots_adjust( 
    left  = 0.0001,  # the left side of the subplots of the figure
    right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    bottom = 0.12,   # the bottom of the subplots of the figure
    top = 0.99,      # the top of the subplots of the figure
    wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    hspace = 0.075)   # the amount of height reserved for white space between subplots

    fig.show()
    # fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/perfil.png', dpi=1200)

# Grafico lineal:
def transforma(delta_z, ddelta_z, longitud_de_onda, dlongitud_de_onda,  cuchilla_pared, dcuchilla_pared):
    '''
    INPUT:
    ancho entre mínimmos del patron, longitud de onda del laser y la distancia de la cuchilla a la pared (cc errores).
    OUTPUT:
    tuple: Apertura y error de la apertura de la barra
    '''
    auxiliar = propagacion_errores({'variables':[('longitud_de_onda', longitud_de_onda, dlongitud_de_onda),
     ('delta_z', delta_z, ddelta_z), ('cuchilla_pared', cuchilla_pared, dcuchilla_pared)], 
    'expr': ('a', '(cuchilla_pared*longitud_de_onda)/delta_z')})
    auxiliar.fit()
    return auxiliar.valor, auxiliar.error
mediciones = pd.read_csv('C:/repos/labo_4/YOUNG_ESTATICO/mediciones_grana_angular.csv', converters={"Mediciones(valor, error)[cm]": ast.literal_eval})
# mediciones = pd.read_csv('C:/repos/labo_4/YOUNG_ESTATICO/mediciones_grana_angular_2.csv', converters={"Mediciones(valor, error)[cm]": ast.literal_eval})
mediciones['Masa[g]'] = [imagenes_2_grana[str(im)] for im in mediciones.Foto]
mediciones['Error Masa[g]'] = np.full(len(imagenes_2_grana.values()), 0.0002)
mediciones.iloc[0,2] = -1
mediciones.sort_values('Masa[g]')#, inplace=True)
mediciones.iloc[0,2] = 'fondo'

# Datos relevantes
L, dL = (28.8+1)/100, np.sqrt(.005**2 + .002**2) # error np.sqrt(.005**2 + .002**2) = 0.005385164807134505 # en metros 
diametro_L, ddiametro_L = 6/1000, .05/1000 # error en metros: 0.05/1000
agarre, dagarre = (28.8 + .5)/100, np.sqrt(.005**2 + .002**2) # error; np.sqrt(.005**2 + .002**2) = 0.005385164807134505 # en metros
l_onda, dl_onda = 670/1e9, 0 # en metros
g, dg = 9.796852, 0.000001 # en metros/seg^2
auxiliar = propagacion_errores({'variables': [('diametro_L', diametro_L, ddiametro_L)],'expr':('I', '(np.pi*diametro_L**4)/64')})
auxiliar.fit()
I, dI = auxiliar.valor, auxiliar.error # en metros^4dm = 0.0002/10000 # kg

masas = []
eje_y = []
eje_y_error = []
apertura_en_reposo, dapertura_en_reposo = transforma(delta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][0]/100, ddelta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][1]/100, longitud_de_onda = l_onda, dlongitud_de_onda= dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
for ind in mediciones.index:
    if ind != 0: # esquivo la medición de calibracion
        m, dm  = mediciones.iloc[ind]['Masa[g]']/1000, mediciones.iloc[ind]['Error Masa[g]']/1000
        masas.append(m)
        deltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][0]/100
        ddeltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][1]/100
        d_m, dd_m = transforma(deltaz, ddeltaz, l_onda, dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
        d, dd = d_m - apertura_en_reposo, np.sqrt(dd_m**2 + dapertura_en_reposo**2)
        y, dy = propagacion_errores(
            {'variables':[('I', I, dI), ('g', g, dg) , ('L', L, dL), ('d', d, dd), ('x', agarre, dagarre)],
             'expr':('dato', '(I/g) * d * (1/( (L*x**2)/2 - (x**3)/6 ))')}).fit()
        eje_y.append(y)
        eje_y_error.append(dy)
error_masas = [.0002/1000 for m in masas]
cov_error = np.diag(np.array(eje_y_error)**2)
reg = regresion_lineal(np.array(masas), np.array(eje_y), cov_y = cov_error, ordenada = True)
reg.fit()
ordenada, pendiente, cov = reg.parametros[0], reg.parametros[1], reg.cov_parametros
v11, v12, v21, v22 = cov[0][0], cov[1][0], cov[0][1], cov[1][1] 
x = np.linspace(masas[0]-.001, masas[-1]+.001, 10000)
ajuste = ordenada + pendiente*x 
# =================================================================================================
# Esto está en el tp 3 de MEFE. Sale de calcular la covarianza para y_predicho usando los datos
# del ajuste (y = b + a.x):
#   Var(y_p, y_p) = der(y_p, b).der(y_p, b).Var(b) + der(y_p, a).der(y_p, a).Var(a) + 
#   2.der(y_p, b).der(y_p, a).Cov(b, a) = Var(b) + Var(a).x**2 + 2.x.Cov(a, b)
# -----> sigma_y = np.sqrt(Var(b) + Var(a).x**2 + 2.x.Cov(a, b))
# =================================================================================================
franja_error = np.sqrt(v11 + v22*x**2 + 2*v12*x)
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 5))
    ax.plot(x, ajuste,  color = 'red', label = 'Ajuste')
    ax.plot(x, ajuste + franja_error,
               '-.', color = 'green', 
               label = 'Error del ajuste')
    ax.plot(x, ajuste - franja_error,
               '-.', color = 'green')
    ax.fill_between(x, ajuste - franja_error,
                       ajuste + franja_error, 
                       facecolor = "gray", alpha = 0.5)
    ax.scatter(masas, eje_y, marker = '.', color = 'k', label = 'Datos')
    ax.errorbar(masas, eje_y, marker = '.', yerr = eje_y_error, fmt = 'none', capsize = 2, color = 'black', label = 'Error de los datos')
    ax.set_xlabel('Masa [kg]', fontsize = 13)
    ax.set_ylabel(r'$\frac{z\, I}{g\, (L\, x^{2}/2 - x^{3}/3)}$[$m \, s^{2}$]', fontsize = 13)
    ax.legend(fontsize = 11, loc = 'best')
    fig.tight_layout()
    # fig.subplots_adjust( 
    # left  = 0.12,  # the left side of the subplots of the figure
    # right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    # bottom = 0.05,   # the bottom of the subplots of the figure
    # top = 0.93,      # the top of the subplots of the figure
    # wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    # hspace = 0.075) 
fig.show()
# fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/lineal.png', dpi = 1200)
# Calculo el modulo de Young
inversa_E, dinversa_E = pendiente, np.sqrt(v22)
E, dE = propagacion_errores(data = {'variables': [('inversa_E', inversa_E, dinversa_E)], 'expr': ('E', '1/inversa_E')}).fit()
E, dE = E/1e9, dE/1e9
print(r'El valor obtenido para el módulo de Young es ($' + r'{}'.format(E) + r' \pm ' + r'{}'.format(dE) + r'$)')

reg.bondad()
reg.r[1][0]
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')
inversa_E*1e9, dinversa_E*1e9

####################################

masas = []
eje_y = []
eje_y_error = []
apertura_en_reposo, dapertura_en_reposo = transforma(delta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][0]/100, ddelta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][1]/100, longitud_de_onda = l_onda, dlongitud_de_onda= dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
for ind in mediciones.index:
    if ind != 0: # esquivo la medición de calibracion
        m, dm  = mediciones.iloc[ind]['Masa[g]']/1000, mediciones.iloc[ind]['Error Masa[g]']/1000
        masas.append(m)
        deltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][0]/100
        ddeltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][1]/100
        d_m, dd_m = transforma(deltaz, ddeltaz, l_onda, dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
        d, dd = d_m - apertura_en_reposo, np.sqrt(dd_m**2 + dapertura_en_reposo**2)
        y, dy = propagacion_errores(
            {'variables':[('I', I, dI), ('g', g, dg) , ('L', L, dL), ('d', d, dd), ('x', agarre, dagarre)],
             'expr':('dato', '(I/g) * d * (1/( (L*x**2)/2 - (x**3)/6 ))')}).fit()
        eje_y.append(y)
        eje_y_error.append(dy)
error_masas = [.0002/1000 for m in masas]
cov_error = np.diag(np.array(error_masas)**2)
reg = regresion_lineal(np.array(eje_y), np.array(masas), cov_y = cov_error, ordenada = True)
reg.fit()
ordenada, pendiente, cov = reg.parametros[0], reg.parametros[1], reg.cov_parametros
v11, v12, v21, v22 = cov[0][0], cov[1][0], cov[0][1], cov[1][1] 
x = np.linspace(eje_y[0], eje_y[-1], 10000)
ajuste = ordenada + pendiente*x 
# =================================================================================================
# Esto está en el tp 3 de MEFE. Sale de calcular la covarianza para y_predicho usando los datos
# del ajuste (y = b + a.x):
#   Var(y_p, y_p) = der(y_p, b).der(y_p, b).Var(b) + der(y_p, a).der(y_p, a).Var(a) + 
#   2.der(y_p, b).der(y_p, a).Cov(b, a) = Var(b) + Var(a).x**2 + 2.x.Cov(a, b)
# -----> sigma_y = np.sqrt(Var(b) + Var(a).x**2 + 2.x.Cov(a, b))
# =================================================================================================
franja_error = np.sqrt(v11 + v22*x**2 + 2*v12*x)
with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 5))
    ax.plot(x, ajuste,  color = 'red', label = 'Ajuste')
    ax.plot(x, ajuste + franja_error,
               '-.', color = 'green', 
               label = 'Error del ajuste')
    ax.plot(x, ajuste - franja_error,
               '-.', color = 'green')
    ax.fill_between(x, ajuste - franja_error,
                       ajuste + franja_error, 
                       facecolor = "gray", alpha = 0.5)
    ax.scatter(eje_y, masas, marker = '.', color = 'k', label = 'Datos')
    ax.errorbar(eje_y, masas, marker = '.', yerr = error_masas, fmt = 'none', capsize = 2, color = 'black', label = 'Error de los datos')
    ax.set_xlabel(r'$\frac{z\, I}{g\, (L\, x^{2}/2 - x^{3}/3)}$[$m \, s^{2}$]', fontsize = 13)
    ax.set_ylabel('Masa [kg]', fontsize = 13)
    ax.legend(fontsize = 11, loc = 'best')
    fig.tight_layout()
    # fig.subplots_adjust( 
    # left  = 0.12,  # the left side of the subplots of the figure
    # right = 0.99,    # the right side of the subplots of the figure, as a fraction of the figure width
    # bottom = 0.05,   # the bottom of the subplots of the figure
    # top = 0.93,      # the top of the subplots of the figure
    # wspace = 0.00015,   # the amount of width reserved for blank space between subplots
    # hspace = 0.075) 
fig.show()
# fig.savefig('C:/repos/labo_4/YOUNG_ESTATICO/Imagenes_informe/lineal.png', dpi = 1200)
# Calculo el modulo de Young
inversa_E, dinversa_E = pendiente, np.sqrt(v22)
E, dE = propagacion_errores(data = {'variables': [('inversa_E', inversa_E, dinversa_E)], 'expr': ('E', '1/inversa_E')}).fit()
E, dE = E/1e9, dE/1e9
print(r'El valor obtenido para el módulo de Young es ($' + r'{}'.format(E) + r' \pm ' + r'{}'.format(dE) + r'$)')

reg.bondad()
reg.r[1][0]
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')
inversa_E*1e9, dinversa_E*1e9
inversa_E/1e9
dinversa_E/1e9