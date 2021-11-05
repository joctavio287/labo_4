import numpy as np, matplotlib.pyplot as plt, pandas as pd, os
os.chdir('C:/repos/labo_4/')

from funciones import *

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

mediciones = pd.read_csv('C:/repos/labo_4/YOUNG_ESTATICO/mediciones_grana_angular.csv')
mediciones['Masa[g]'] = list(imagenes_2_grana.values())
mediciones['Error Masa[g]'] = np.full(len(imagenes_2_grana.values()), 0.0002)
mediciones['Mediciones(valor, error)[cm]'] = mediciones['Mediciones(valor, error)[cm]'].apply(lambda x: tuple(float(y) for y in x.split(', ')))

# Magnitudes relevantes:
L, dL = (28.8+1)/100, np.sqrt(.005**2 + .002**2) # error np.sqrt(.005**2 + .002**2) = 0.005385164807134505 # en metros 
diametro_L, ddiametro_L = 6/1000, .05/1000 # error en metros: 0.05/1000
agarre, dagarre = (28.8 + .5)/100, np.sqrt(.005**2 + .002**2) # error; np.sqrt(.005**2 + .002**2) = 0.005385164807134505 # en metros
l_onda, dl_onda = 670/1e9, 0 # en metros
g, dg = 9.876, 0 # en metros/seg^2
auxiliar = propagacion_errores({'variables': [('diametro_L', diametro_L, ddiametro_L)],'expr':('I', '(np.pi*diametro_L**4)/64')})
auxiliar.fit()
I, dI = auxiliar.valor, auxiliar.error # en metros^4

# Seteo el cero de la medicion:
apertura_en_reposo, dapertura_en_reposo = transforma(delta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][0]/100, ddelta_z = mediciones.iloc[1]['Mediciones(valor, error)[cm]'][1]/100, 
longitud_de_onda = l_onda, dlongitud_de_onda= dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )

# Armo una lista con todas las mediciones para después tomar el promedio:
modulos_y, dmodulos_y = [], []
for ind in mediciones.index:
    if mediciones.loc[ind]['Masa[g]'] != 'fondo' and mediciones.loc[ind]['Masa[g]'] != 0:
        # Transformo las mediciones a kg y metros para suplantar en la fórmula:
        deltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][0]/100
        ddeltaz = mediciones.loc[ind]['Mediciones(valor, error)[cm]'][1]/100
        d_m, dd_m = transforma(deltaz, ddeltaz, l_onda, dl_onda, cuchilla_pared = 153.6/100, dcuchilla_pared=.6/100 )
        d, dd = d_m - apertura_en_reposo, np.sqrt(dd_m**2 + dapertura_en_reposo**2)
        masa, dmasa = mediciones.loc[ind]['Masa[g]']/1000, mediciones.loc[ind]['Error Masa[g]']/1000
        aux = propagacion_errores({'variables': [('masa', masa, dmasa), ('g', g, dg), ('L', L, dL), ('agarre', agarre, dagarre/100), ('I', I, dI/100), ('d', d, dd)],
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
E, dE
