import time, pyvisa, numpy as np, matplotlib.pyplot as plt, pandas as pd

# ========================================================================
#                         Funciones auxiliares
# ========================================================================

# Función para sacarle una foto a la pantalla del osciloscopio por canal
def medir(inst, channel = 1):
    """
    Adquiere los datos del canal canal especificado.
    WFMPRE:XZEro? query the time of first data point in waveform
          :XINcr? query the horizontal sampling interval
          :YZEro? query the waveform conversion factor
          :YMUlt? query the vertical scale factor
          :YOFf?  query the vertical position
    INPUT: 
    -inst --> objeto de pyvisa: el nombre de la instancia.
    -channel --> int: canal del osciloscopio que se quiere medir.
    OUTPUT:
    Tupla de dos arrays de numpy: el tiempo y la tensión.
    """
    inst.write('DATa:SOUrce CH' + str(channel))
    xze, xin, yze, ymu, yoff = inst.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?', separator = ';')
    datos = inst.query_binary_values('CURV?', datatype = 'B', container = np.array)
    data = (datos - yoff)*ymu + yze
    tiempo = xze + np.arange(len(data)) * xin
    return tiempo, data

# Función para integrar señales
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

# Hacemos la conversión de resistencia [Ohms] a Temperatura [C]
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

# ========================================================================
# Chequeamos los instrumentos que están conectados
# ========================================================================

rm = pyvisa.ResourceManager()
instrumentos = rm.list_resources()

# Hay que chequear el numero de la lista para asignarlo correctamente:
osc = rm.open_resource(instrumentos[0])
mult = rm.open_resource(instrumentos[1])
# ========================================================================
# Vamos a tomar mediciones indirectas de la temperatura del núcleo del 
# transformador, recopilando la impedancia de una resistencia adosada 
# al mismo. Además tomamos mediciones de la tensión del primario y el 
# secundario. 
# ========================================================================

# La graduación vertical (1 cuadradito) puede ser: [50e-3, 100e-3, 200e-3, 500e-3, .1e1, .2e1, .5e1]
# Medimos con diferente escalas los dos canales y estas cambiaron para el experimento con y sin integrador
escala = .2e1
osc.write(f'CH{1}:SCale {escala:3.1E}')
osc.write(f'CH{2}:SCale {escala:3.1E}')

# ========================================================================
# Setemos la escala horizontal teniendo en cuenta que la frecuencia del toma
# (220 V) es de 50 Hz.
# ========================================================================
freq = 50 # Hz
escala_t = 4/(10*2*np.pi*freq) # para que entren cuatro períodos en la pantalla (tiene diez cuadraditos, x eso el 10)
osc.write(f'HORizontal:MAin:SCale {escala_t:3.1E}')

# ========================================================================
# Como el código en Python es secuencial vamos a tener que medir en tiempos
# diferidos las tres magnitudes y después interpolar los datos para 'hacer
# que coincidan'.
# ========================================================================

# Las mediciones se van a efectuar cada 'intervalo_temporal'
intervalo_temporal = 4

# Hay que hacer una iteración para saber cuánto tarda y podamos asignar bien el intervalo temporal
t_auxiliar_1 = time.time()

# ESTA ES UNA ITERACION PELADA
t = time.time()
float(mult.query('MEASURE:FRES?'))
marca_resistencia = t + (time.time() - t)/2
marca_resistencia-t
t = time.time()
medir(osc, 1)
marca_tension_1 = t + (time.time() - t)/2
marca_tension_1 - t
t = time.time()
medir(osc, 2)
marca_tension_2 = t + (time.time() - t)/2
marca_tension_2 - t

t_auxiliar_2 = time.time()

# Actualizamos el valor del intervalo temporal (resto de manera tal que el tiempo del inervalo 
# resultante sea, en verdad, el especificado más arriba):
intervalo_temporal -= t_auxiliar_2 - t_auxiliar_1

# Asumimos que el fenómeno dura 3.5'= 210'', modificar de ser necesario
tiempo_total = 150

# Creo las iteraciones (tiempo_total/intervalo_temporal es el número de pasos que vamos a tomar)
iteraciones = np.arange(0, int(tiempo_total/intervalo_temporal), 1)

# Las tres magnitudes que vamos a medir
resistencia, tension_1, tension_2 = [], [], []

# Las marcas temporales de las mediciones
marca_temporal_resistencia, marca_temporal_tension_1, marca_temporal_tension_2 = [], [], []

# Tomamos la referencia del tiempo inicial
t_0 = time.time()

# Hacemos iteraciones
for i in iteraciones:

    # Medición de resistencia
    t = time.time()
    resistencia.append(float(mult.query('MEASURE:FRES?')))
    # La marca temporal es el promedio entre antes y después de medir
    marca_resistencia = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_resistencia.append(marca_resistencia-t_0)

    # Medición de tensión en el primario
    t = time.time()
    tension_1.append(medir(osc, 1))
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_1 = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_tension_1.append(marca_tension_1 - t_0)

    # Medición de tensión en el secundario
    t = time.time()
    tension_1.append(medir(osc, 2))
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_2 = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_tension_2.append(marca_tension_2 - t_0)
    
    # Intervalo temporal entre mediciones
    time.sleep(intervalo_temporal)

# ESTABA MAL# Corregimos desfasajes temporales entre la tensión del primario y el secundario
# for i in iteraciones:
#     # Este es el desfasaje entre que sacamos la captura del primario al secundario
#     desfasaje = marca_temporal_tension_2[i] - marca_temporal_tension_1[i]
#     # ATENCION ESTA INTERPOLACION ES INCORRECTA MOSTRARLE EL DIBU A SOFI
#     # Chequear que lo que viene del osc es arrays, dado el caso contrario cambiarlos
#     tension_2[i] = tension_1[i][0], np.interp(tension_1[i][0], tension_2[i][0] - desfasaje, tension_2[i][1])


# Convertimos en formato numpy los datos
resistencia = np.array(resistencia)
marca_temporal_resistencia,marca_temporal_tension_1,marca_temporal_tension_2 = np.array(marca_temporal_resistencia),np.array(marca_temporal_tension_1),np.array(marca_temporal_tension_2)

# Antes de analizar lo medido, guardamos los datos
datos = pd.DataFrame(data = {'resistencia' : resistencia, 'marca_temporal_resistencia' : marca_temporal_resistencia,
                             'tension_1': tension_1, 'marca_temporal_tension_1' : marca_temporal_tension_1,
                             'tension_2': tension_2, 'marca_temporal_tension_2' : marca_temporal_tension_2})

# Modificar acorde a dónde guardemos en la compu del labo:
full_path = 'C:/repos/labo_4/FERROMAGNETISMO/Mediciones/'

# Cambiar después de cada medición, sino los datos se van a reescribir
o = 0
datos.to_csv(full_path + 'medicion_{}'.format(o))
datos.to_csv(full_path + 'medicion_sin_integrador_{}'.format(o))

#################################### ANALISIS DE DATOS ####################################

# =================================================================================================
# Necesitamos convertir los datos de la resistencia a temperatura, para ello nos valemos de la fun_
# ción definida al comienzo del código
# =================================================================================================

# ATENCION NO PRINTEAR 'temperaturas_auxiliar', ni 'conversor', son muchos números y rompen la maquina

# Un rango de temperaturas muy fino, se puede hacer todavía más fino si la resolución de las mediciones es muy buena
temperaturas_auxiliar = np.linspace(-300, 300, 100000) 

# Creamod un diccionario para transformar resistencias en temperaturas
conversor = {r: t for r, t in zip(funcion_conversora_temp(temperaturas_auxiliar), temperaturas_auxiliar)}

# Transformamos las impedancias medidas [Ohms] a temperatura [K]
temperatura = [conversor[min(conversor.keys(), key = lambda x:abs(x-r))] + 273.15 for r in resistencia]
marca_temporal_temperatura = marca_temporal_resistencia.copy()

# =================================================================================================
# De alguna manera tenemos que unir las mediciones para que todas esten asociadas al mismo tiempo,
# para ello podemos tomar alguna de las mediciones (temperatura, tension_primario o secundario) y 
# utilizar su huella temporal como referencia a partir de la cual interpolaremos el resto.
# Tomamos la tensión del primario como referencia (después cambiar como sea conveniente).
# =================================================================================================

# Interpolación de la temperatura en base a la huella temporal de la tension del primario

temperatura = np.interp(marca_temporal_tension_1, marca_temporal_temperatura, temperatura)
error_temporal_temperatura = np.abs(marca_temporal_tension_1 - marca_temporal_temperatura)

# =================================================================================================
# Antes de graficar sólo resta integrar numéricamente la tensión del secundario para cada 
# una de las mediciones:
# =================================================================================================

# # Si la señal tiene offset
# tension_1_interpolada -= tension_1_interpolada.mean()

# Realizo la integral:

# for i in iteraciones:
#     tension_2[i] = tension_2[i][0], funcion_integradora(tension_2[i][1])

# =================================================================================================
# Ahora que tenemos las tres mediciones asociadas al mismo tiempo podemos graficar los datos:
# =================================================================================================

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,7))
    # Graficamos la tensión integrada del secundario (proporcional a B) en función de la tensión del primario (proporcional a H)
    # ax.plot(tension_2_interpolada_integrada, tension_2_interpolada, label = 'Curva de histeresis')
    ax.set_xlabel(r'\propto B [UNIDADES?]')
    ax.set_ylabel(r'\propto H [V]')
    fig.show()


# # Modelos de juguete

# # Para integrar

# # Tiempo total
# T = 10*np.pi

# # El factor de escala está basado en algunas simulaciones que corrí
# numero_de_puntos = int(T/0.15) # Este es un valor que da una buena integral

# # Datos sintéticos
# x = np.linspace(1, T, numero_de_puntos)
# y = 3*np.sin(x) + 5
# # y = 1/x

# # Si los datos tienen media
# y -= y.mean()

# # Calculo integral numérica
# integral_y = np.cumsum(y) * (T/len(x))  #- (y[-1] - y[0])
# integral_real_y = -3*np.cos(x) 
# # integral_real_y = np.log(x)

# # Grafico
# plt.figure()
# plt.plot(x, integral_y, color = 'black', label = 'Integral numérica')
# plt.plot(x, integral_real_y, color = 'violet', label = 'Integral real')
# plt.plot(x, y, color = 'red', label = 'Funcion real')
# plt.legend()
# plt.show(block = False)
