import time, pyvisa, numpy as np, matplotlib.pyplot as plt, pandas as pd

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
escala = .2e1
osc.write(f'CH{1}:SCale {escala:3.1E}')
osc.write(f'CH{2}:SCale {escala:3.1E}')

# ========================================================================
# Setemos la escala horizontal teniendo en cuenta que la frecuencia del toma
# (220 V) es de 50 Hz.
# ========================================================================
freq = 50 # Hz
escala_t = (4/10)*(1/freq) # para que entren cuatro períodos en la pantalla
osc.write(f'HORizontal:MAin:SCale {escala_t:3.1E}')

# ========================================================================
# Como el código en Python es secuencial vamos a tener que medir en tiempos
# diferidos las tres magnitudes y después interpolar los datos para 'hacer
# que coincidan'.
# ========================================================================

# Las tres magnitudes que vamos a medir
resistencia, tension_1, tension_2 = [], [], []

# Las marcas temporales de las mediciones
marca_temporal_resistencia, marca_temporal_tension_1, marca_temporal_tension_2 = [], [], []

# Las mediciones se van a efectuar cada 'intervalo_temporal'
intervalo_temporal = 5

# Hay que hacer una iteración para saber cuánto tarda y podamos asignar bien el intervalo temporal
t_auxiliar_1 = time.time()

# ESTA ES UNA ITERACION PELADA
t = time.time()
float(mult.query('MEASURE:FRES?'))
t + (time.time() - t)/2
t- t + (time.time() - t)/2
t = time.time()
osc.write('MEASUrement:IMMed:SOU CH1; TYPe PK2k')
osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0]
marca_tension_1 = t + (time.time() - t)/2
marca_tension_1
t = time.time()
osc.write('MEASUrement:IMMed:SOU CH2; TYPe PK2k')
osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0]
marca_tension_2 = t + (time.time() - t)/2
t - marca_tension_2
t_auxiliar_2 = time.time()

# Actualizamos el valor del intervalo temporal (resto de manera tal que el tiempo del inervalo 
# resultante sea, en verdad, el especificado más arriba):
intervalo_temporal -= t_auxiliar_2 - t_auxiliar_1

# Asumimos que el fenómeno dura 3.5'= 210'', modificar de ser necesario
tiempo_total = 210 

# Creo las iteraciones (tiempo_total/intervalo_temporal es el número de pasos que vamos a tomar)
iteraciones = np.arange(0, tiempo_total/intervalo_temporal, 1)

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
    marca_temporal_resistencia.append(t_0 - marca_resistencia)

    # Medición de tensión en el primario
    t = time.time()
    osc.write('MEASUrement:IMMed:SOU CH1; TYPe PK2k')
    tension_1.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_1 = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_tension_1.append(t_0 - marca_tension_1)

    # Medición de tensión en el secundario
    t = time.time()
    osc.write('MEASUrement:IMMed:SOU CH2; TYPe PK2k')
    tension_2.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_2 = t + (time.time() - t)/2
    # Appendeamos el tiempo respecto al t_0
    marca_temporal_tension_2.append(t_0 - marca_tension_2)
    
    # Intervalo temporal entre mediciones
    time.sleep(intervalo_temporal)

# Convertimos en formato numpy los datos
resistencia,tension_1,tension_2=np.array(resistencia),np.array(tension_1),np.array(tension_2)
marca_temporal_resistencia,marca_temporal_tension_1,marca_temporal_tension_2=np.array(marca_temporal_resistencia),np.array(marca_temporal_tension_1),np.array(marca_temporal_tension_2)

# Antes de analizar lo medido, guardamos los datos
datos = pd.Dataframe(data = {'resistencia[Ohms]':resistencia, 'marca_temporal_resistencia' : marca_temporal_resistencia,
                             'tension_1[V]': tension_1, 'marca_temporal_tension_1' : marca_temporal_tension_1,
                             'tension_2[V]': tension_2, 'marca_temporal_tension_2' : marca_temporal_tension_2})

# Modificar acorde a dónde guardemos en la compu del labo:
full_path = 'C:/repos/labo_4/FERROMAGNETISMO/Mediciones/'

# Cambiar después de cada medición, sino los datos se van a reescribir
o = 0
datos.to_csv(full_path + 'medicion_{}'.format(o))

#################################### ANALISIS DE DATOS ####################################

# Hacemos la conversión de resistencia [Ohms] a Temperatura [C]
def funcion_conversora(t):
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

# Para ver cómo es la conversion
plt.figure()
plt.plot(np.linspace(-200,800,1000), funcion_conversora(np.linspace(-200,800,1000)))
plt.xlabel('Temeperatura [C]')
plt.ylabel('Resistencia [Ohms]')
plt.grid(True)
plt.legend()
plt.show(block = False)

# ATENCION NO PRINTEAR NI 'temperaturas_auxiliar', ni 'conversor', son muchos numeros y rompen la maquina.

# Un rango de temperaturas muy fino, se puede hacer todavía más fino si la resolución de las mediciones es muy buena.
temperaturas_auxiliar = np.linspace(-300, 300, 100000) 

# Creamod un diccionario para transformar resistencias en temperaturas:
conversor = {r: t for r, t in zip(funcion_conversora(temperaturas_auxiliar), temperaturas_auxiliar)}

# Transformamos las impedancias medidas [Ohms] a temperatura [K]
temperatura = [conversor[min(conversor.keys(), key = lambda x:abs(x-r)) + 273.15] for r in resistencia]
marca_temporal_temperatura = marca_temporal_resistencia

# =================================================================================================
# De alguna manera tenemos que unir las mediciones para que todas esten asociadas al mismo tiempo,
# para ello podemos tomar alguna de las mediciones (temperatura, tension_primario o secundario) co_
# mo y utilizar su huella temporal como referencia a partir de la cual interpolaremos el resto.
# Tomamos la temperatura como referencia (después cambiar como sea conveniente).
# =================================================================================================

# Interpolación de la tension_1 y la tension_2 en base a la huella temporal de la temperatura
tension_1_interpolada = np.interp(marca_temporal_temperatura, marca_temporal_tension_1, tension_1)
error_temporal_tension_1 =  np.abs(marca_temporal_tension_1 - marca_temporal_temperatura)

tension_2_interpolada = np.interp(marca_temporal_temperatura, marca_temporal_tension_2, tension_2)
error_temporal_tension_2 =  np.abs(marca_temporal_tension_2 - marca_temporal_temperatura)


# =================================================================================================
# Antes de graficar sólo resta integrar numéricamente la tensión del secundario:
# =================================================================================================

# # Si la señal tiene offset
# tension_1_interpolada = tension_1_interpolada - tension_1_interpolada.mean()

# Realizo la integral:
tension_2_interpolada_integrada = np.cumsum(tension_2_interpolada)

# =================================================================================================
# Ahora que tenemos las tres mediciones asociadas al mismo tiempo podemos graficar los datos:
# =================================================================================================

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,7))
    # Graficamos la tensión integrada del secundario (proporcional a B) en función de la tensión del primario (proporcional a H)
    ax.plot(tension_2_interpolada_integrada, tension_2_interpolada, label = 'Curva de histeresis')
    ax.set_xlabel(r'\propto B [UNIDADES?]')
    ax.set_ylabel(r'\propto H [V]')
    fig.show()



# # Modelos de juguete

# # Para interpolar
# x = np.linspace(0, 2, 10)
# y = np.sin(x)
# xvals = np.linspace(0, 2, 3)
# yinterp = np.interp(xvals, x, y)
# plt.figure()
# plt.scatter(xvals, yinterp, color = 'red', s = 5, label = 'Valores interpolados')
# plt.scatter(x, y, color = 'black', s = 5, label = 'Valores reales')
# plt.legend()
# plt.show(block=False)

# Para integrar

# Tiempo total
T = 2*np.pi

# El factor de escala está basado en algunas simulaciones que corrí
numero_de_puntos = int(T/0.15) # Este es un valor que da una buena integral

# Datos sintéticos
x = np.linspace(0.1, T, numero_de_puntos)
y = np.sin(x) 
# y = 1/x

# Si los datos tienen media
# y -= y.mean()

# Calculo integral numérica
integral_y = np.cumsum(y) * (T/len(x)) #- (y[-1] - y[0])
integral_real_y = -np.cos(x) 
# integral_real_y = np.log(x)

# Grafico
plt.figure()
plt.plot(x, integral_y, color = 'black', label = 'Integral numérica')
plt.plot(x, integral_real_y, color = 'violet', label = 'Integral real')
plt.plot(x, y, color = 'red', label = 'Funcion real')
plt.legend()
plt.show(block = False)


x = np.linspace(0, T, numero_de_puntos)
y = np.sin(x) 
integral_y = np.cumsum(y) * (T/len(x)) #- (y[-1] - y[0])
integral_real_y = np.cos(x) 
