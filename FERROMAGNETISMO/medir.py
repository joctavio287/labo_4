import time, pyvisa, numpy as np, matplotlib.pyplot as plt

# ========================================================================
# Chequeo los instrumentos que están conectados
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
# Seteo la escala horizontal teniendo en cuenta que la frecuencia del toma
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

# Las tres magnitudes que vamos a medir:
resistencia, tension_1, tension_2 = [], [], []

# Las marcas temporales de las mediciones:
marca_temporal_resistencia, marca_temporal_tension_1, marca_temporal_tension_2 = [], [], []

# Las mediciones se van a efectuar cada 'intervalo_temporal' s:
intervalo_temporal = 5

# Hay que hacer una iteración para saber cuánto tarda y podamos asignar bien el intervalo temporal:
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

# Actualizo el valor del intervalo temporal (resto de manera tal que el tiempo del inervalo 
# resultante sea el especificado más arriba):
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
    # Appendeo el tiempo respecto al t_0
    marca_temporal_resistencia.append(t_0 - marca_resistencia)

    # Medición de tensión en el primario
    t = time.time()
    osc.write('MEASUrement:IMMed:SOU CH1; TYPe PK2k')
    tension_1.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_1 = t + (time.time() - t)/2
    # Appendeo el tiempo respecto al t_0
    marca_temporal_tension_1.append(t_0 - marca_tension_1)

    # Medición de tensión en el secundario
    t = time.time()
    osc.write('MEASUrement:IMMed:SOU CH2; TYPe PK2k')
    tension_2.append(osc.query_ascii_values('MEASUrement:IMMed:VALue?')[0])
    # La marca temporal es el promedio entre antes y después de medir
    marca_tension_2 = t + (time.time() - t)/2
    # Appendeo el tiempo respecto al t_0
    marca_temporal_tension_2.append(t_0 - marca_tension_2)
    
    # Intervalo temporal entre mediciones
    time.sleep(intervalo_temporal)

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
temperaturas_auxiliar = np.linspace(-300, 300, 20000) # Un rango de temperaturas muy fino
# Creo un diccionario para transformar resistencias en temperaturas:
conversor = {r: t for r, t in zip(funcion_conversora(temperaturas_auxiliar), temperaturas_auxiliar)}

# Transformamos las impedancias medidas [Ohms] a temperatura [K]
temperatura = [conversor[min(conversor.keys(), key = lambda x:abs(x-r)) + 273.15] for r in resistencia]


# Unimos los datos, para eso interpolamos usando la temperatura como referencia
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xvals = np.linspace(0, 2*np.pi, 50)
yinterp = np.interp(xvals, x, y)