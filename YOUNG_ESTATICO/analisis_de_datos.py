import numpy as np, matplotlib.pyplot as plt, pandas as pd, os
from funciones import *
import ast

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

masas_2 = {'base':14.6624,1:2.1160,2:2.0399,3:1.6966,4:10.046,5:2.1535,6:5.0952,7:20.0350,10:4.7474,11:9.9163,12:2.1593,15:2.1468}

# Imagenes sacadas con grana angular:
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
mediciones = pd.read_csv('C:/repos/labo_4/YOUNG_ESTATICO/mediciones_grana_angular.csv', converters={"Mediciones(valor, error)[cm]": ast.literal_eval})
# mediciones = pd.read_csv('C:/repos/labo_4/YOUNG_ESTATICO/mediciones_grana_angular_2.csv', converters={"Mediciones(valor, error)[cm]": ast.literal_eval})
mediciones['Masa[g]'] = [imagenes_2_grana[str(im)] for im in mediciones.Foto]
mediciones['Error Masa[g]'] = np.full(len(imagenes_2_grana.values()), 0.0002)

# Cambio este valor porque es un string y no me permite ordenar strings:
mediciones.iloc[0,2] = -1

# Ordeno de menor a mayor acorde a la columna de las masas:
mediciones.sort_values('Masa[g]')#, inplace=True)

# Reasigno el nombre al valor cambiado para ordenar:
mediciones.iloc[0,2] = 'fondo'

#DISTANCIAS:
#    CUCHILA-PTOFIJO: 28.8+-.5
#    PANTALLA-CUCHILLA: 153.6 +-.6
#    PANTALLA-CELULAR: 21.5 +- .2
#    CUCHILLA-BASE: .5+-.2
#    LARGO VARILLLA: 52+-.2
#MASAS:
#    CUCHILLA SUELTA (UN POCO MAS LARGA Q LA QUE ESTABA PEGADA ): 3.777 gramos, 6CM +- .1 de largo 
#    LARGO CUCHILLA DE LA VARILLA: 4+-.1 cm
#    CUCHILLA VARILLA LATON: 10 +-.1 cm
#    PESO VARILLA (INCLUYENDO CUCHILLA): 117.9236
# En gramos, el error es 0.0002 g
#LONGITUD DE ONDA DEL LASER: 670nm

# =====================================================================================
# Procedimiento 1: para cada medición utilizo la fórmula y despejo el módulo de Young,
# =====================================================================================

# Magnitudes relevantes:
L, dL = (28.8+1)/100, np.sqrt(.005**2 + .002**2) # error np.sqrt(.005**2 + .002**2) = 0.005385164807134505 # en metros 
diametro_L, ddiametro_L = 6/1000, .05/1000 # error en metros: 0.05/1000
agarre, dagarre = (28.8 + .5)/100, np.sqrt(.005**2 + .002**2) # error; np.sqrt(.005**2 + .002**2) = 0.005385164807134505 # en metros
l_onda, dl_onda = 670/1e9, 0 # en metros
g, dg = 9.796852, 0.000001 # en metros/seg^2
auxiliar = propagacion_errores({'variables': [('diametro_L', diametro_L, ddiametro_L)],'expr':('I', '(np.pi*diametro_L**4)/64')})
auxiliar.fit()
I, dI = auxiliar.valor, auxiliar.error # en metros^4

# Seteo el cero de la medicion:
apertura_en_reposo, dapertura_en_reposo = transforma(delta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][0]/100, ddelta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][1]/100, 
longitud_de_onda = l_onda, dlongitud_de_onda= dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )

# Armo una lista con todas las mediciones para después tomar el promedio:
modulos_y, dmodulos_y = [], []
aperturas, daperturas = [], []
for ind in mediciones.index:
    if mediciones.loc[ind]['Masa[g]'] != 'fondo' and mediciones.loc[ind]['Masa[g]'] != 0:
        # Transformo las mediciones a kg y metros para suplantar en la fórmula:
        deltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][0]/100
        ddeltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][1]/100
        d_m, dd_m = transforma(deltaz, ddeltaz, l_onda, dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
        d, dd = d_m - apertura_en_reposo, np.sqrt(dd_m**2 + dapertura_en_reposo**2)
        aperturas.append(d)
        daperturas.append(dd)
        masa, dmasa = mediciones.loc[ind]['Masa[g]']/1000, mediciones.loc[ind]['Error Masa[g]']/1000
        aux = propagacion_errores({'variables': [('masa', masa, dmasa), ('g', g, dg), ('L', L, dL), ('agarre', agarre, dagarre/100), ('I', I, dI), ('d', d, dd)],
        'expr':('E', '(masa * g * (L*agarre**2 - (agarre**3)/3))/(2*I*d)')})
        aux.fit()
        modulos_y.append(aux.valor)
        dmodulos_y.append(aux.error)

modulos_y, dmodulos_y = np.array(modulos_y), np.array(dmodulos_y)
E = modulos_y.mean()/1e9
dE = 0
for i in dmodulos_y:
    dE += (i/(len(dmodulos_y)))**2
dE = np.sqrt(dE)/1e9

# Los módulos de Young de las mediciones son:
masas_zip = [value for value in imagenes_2_grana.values() if (value != 'fondo'and value!= 0)]
# Las ordeno de mayor a menor:
masas_zip.sort()
texto = f'Masa: ({0} ± {.0002})g:'
texto += f'--> Apertura inicial: ({apertura_en_reposo} ± {dapertura_en_reposo}) m.'
print(texto)
for masa, apertura, dapertura, modulo, dmodulo in zip(masas_zip, aperturas, daperturas, modulos_y, dmodulos_y):
    texto = f'Masa: ({masa} ± {.0002})g:'
    texto += f'--> Apertura respecto a la inicial: ({apertura} ± {dapertura}) m.'
    texto += f'--> Módulo: ({modulo/1e9} ± {dmodulo/1e9}) GPa.\n'
    print(texto)
print(f'El promedio de los resultados nos indica que el modulo de Young del acero inoxidable 304 es ({E} ± {dE}) GPa.')


# =====================================================================================
# Procedimiento 2: hago un gráfico masa vs (I/g) * z * ( (L*x**2)/2 - (x**3)/6 )**-1,
# la pendiente será 1/E.
# =====================================================================================

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

cov_error = np.diag(np.array(eje_y_error)**2)
reg = regresion_lineal(np.array(masas), np.array(eje_y), cov_y = cov_error, ordenada = True)
reg.fit()
ordenada, pendiente, cov = reg.parametros[0], reg.parametros[1], reg.cov_parametros
v11, v12, v21, v22 = cov[0][0], cov[1][0], cov[0][1], cov[1][1] 

# Auxiliares par graficar:
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
    ax.scatter(masas, eje_y, marker = '.', color = 'k', label = 'Datos')
    ax.errorbar(masas, eje_y, marker = '.', yerr = eje_y_error, fmt = 'none', capsize = 2, color = 'black', label = 'Error de los datos')
    ax.set_xlabel('Masa [kg]', fontsize = 13)
    ax.set_ylabel(r'$\frac{z\, I}{g\, (L\, x^{2}/2 - x^{3}/3)}$[$m x s^{2}$]', fontsize = 13)
    ax.legend(fontsize = 11, loc = 'best')
fig.tight_layout()
fig.show()

# Calculo el modulo de Young
inversa_E, dinversa_E = pendiente, np.sqrt(v22)
E, dE = propagacion_errores(data = {'variables': [('inversa_E', inversa_E, dinversa_E)], 'expr': ('E', '1/inversa_E')}).fit()
E, dE = E/1e9, dE/1e9
print(r'El valor obtenido para el módulo de Young es ($' + r'{}'.format(E) + r' \pm ' + r'{}'.format(dE) + r'$)')

reg.bondad()
print(f'El coeficiente de correlación lineal de los datos es: {reg.r[1][0]}')


for ind in mediciones.index:
    if mediciones.loc[ind]['Masa[g]'] != 'fondo' and mediciones.loc[ind]['Masa[g]'] != 0:
        # Transformo las mediciones a kg y metros para suplantar en la fórmula:
        deltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][0]/100
        ddeltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][1]/100
        d_m, dd_m = transforma(deltaz, ddeltaz, l_onda, dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
        d, dd = d_m - apertura_en_reposo, np.sqrt(dd_m**2 + dapertura_en_reposo**2)
        masa, dmasa = mediciones.loc[ind]['Masa[g]']/1000, mediciones.loc[ind]['Error Masa[g]']/1000
        print(masa,d,dd)
        print(deltaz,ddeltaz)

apertura_en_reposo, dapertura_en_reposo 
mediciones.iloc[1]['Mediciones(valor, error)[cm]'][0]/100, mediciones.iloc[1]['Mediciones(valor, error)[cm]'][1]/100
dapertura_en_reposo*1000
